import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from .utils import (
    LayerNorm,
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU
)


# Custom weight initialization function
def _init_weights(m):
    if isinstance(m, MinkowskiConvolution):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiConvolutionTranspose):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiDepthwiseConvolution):
        trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiLinear):
        trunc_normal_(m.linear.weight, std=.02)
        if m.linear.bias is not None:
            nn.init.constant_(m.linear.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Block(nn.Module):
    """ Sparse ConvNeXtV2 Block. 

    Args:
        dim (int): Number of input channels.
        kernel_size (int): Size of input kernel.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=5, drop_path=0., D=3):
        super().__init__()

        self.dwconv = MinkowskiDepthwiseConvolution(dim, kernel_size=kernel_size, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)
        self.act = MinkowskiGELU()
        self.grn = MinkowskiGRN(4  * dim)
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.drop_path = MinkowskiDropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)

        return x


class MinkUNetConvNeXtV2(nn.Module):

    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)
        
        self.ds_steps = args.ds_steps

        """Encoder"""
        depths=[2, 4, 4, 8, 8, 8]
        dims = (16, 32, 64, 128, 256, 512)
        #dims = (16, 32, 64, 96, 96, 96)
        kernel_size = 3
        drop_path_rate = 0.0

        assert len(depths) == len(dims)

        self.nb_elayers = len(dims)
       
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stem = nn.Sequential(
            MinkowskiConvolution(in_channels, dims[0], kernel_size=1, stride=1, dimension=D),
            MinkowskiLayerNorm(dims[0], eps=1e-6),
        ) 

        for i in range(self.nb_elayers):
            encoder_layer = nn.Sequential(
                *[Block(dim=dims[i], kernel_size=kernel_size, drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            cur += depths[i]

            if i < self.nb_elayers - 1:  
                downsample_layer = nn.Sequential(
                    MinkowskiLayerNorm(dims[i], eps=1e-6),                
                    MinkowskiConvolution(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=D),
                )
                self.downsample_layers.append(downsample_layer)

        """Decoder"""
        last_enc_depth = depths[-1]
        depths = [2, 2, 2, 2, 2]
        #depths = depths[:-1][::-1]
        dims = dims[::-1]
        decoder_embed_dim = 32

        self.nb_dlayers = len(dims) - 1

        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(dp_rates[-last_enc_depth], 0, sum(depths))]
        cur = 0

        for i in range(self.nb_dlayers):
            upsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6), 
                MinkowskiConvolutionTranspose(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=D),
            )
            self.upsample_layers.append(upsample_layer)

            decoder_layer = nn.Sequential(
                *[Block(dim=dims[i+1], kernel_size=kernel_size, drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.decoder_layers.append(decoder_layer)
            cur += depths[i]

        """Cls layers"""
        self.cls_layers = nn.ModuleList()
        for i in range(self.nb_dlayers):
            cls_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i+1], eps=1e-6),
                #MinkowskiConvolution(dims[i+1], dims[i+1], kernel_size=1, stride=1, dimension=D),
                Block(dim=dims[i+1], kernel_size=kernel_size, drop_path=0., D=D),
                MinkowskiConvolution(dims[i+1], out_channels, kernel_size=1, stride=1, dimension=D),
            )
            self.cls_layers.append(cls_layer)

        """ Pool just for generating downsampled labels """        
        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3) 

        """ Initialise weights """
        self.apply(_init_weights)

    def forward(self, x, y):
        """ Generate labels for deep supervision """
        ys = []
        for i in range(self.ds_steps):
            if i==0:
                y_aux = y.detach()
            else:
                y_aux = self.pool(y_aux)
            if i < self.ds_steps:
                ys.append(y_aux)
 
        """Encoder"""
        x = self.stem(x)
        x_enc = []
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x_enc.append(x)
                x = self.downsample_layers[i](x)
        x_enc = x_enc[::-1]

        """Decoder"""
        out_cls = []
        for i in range(self.nb_dlayers):
            x = self.upsample_layers[i](x)
            x = x + x_enc[i]
            x = self.decoder_layers[i](x)
            if i >= (self.nb_dlayers - self.ds_steps):
                out_cl = self.cls_layers[i](x)
                out_cls.insert(0, out_cl)

        return out_cls, ys

