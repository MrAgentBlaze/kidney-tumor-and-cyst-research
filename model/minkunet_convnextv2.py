import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
import torch.nn.functional as F
from .utils import (
    Block,
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
    if isinstance(m, ME.MinkowskiConvolution):
        nn.init.trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, ME.MinkowskiGenerativeConvolutionTranspose):
        nn.init.trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, ME.MinkowskiDepthwiseConvolution):
        nn.init.trunc_normal_(m.kernel, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, ME.MinkowskiLinear):
        nn.init.trunc_normal_(m.linear.weight, std=.02)
        if m.linear.bias is not None:
            nn.init.constant_(m.linear.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MinkUNetConvNeXtV2(nn.Module):

    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)
        
        self.contrastive = args.contrastive
        self.finetuning = args.finetuning
        self.ds_steps = args.ds_steps

        """Encoder"""
        depths=[1, 2, 2, 4, 4, 4]
        dims = (16, 32, 64, 96, 96, 96)
        drop_path_rate = 0.0

        assert len(depths) == len(dims)

        self.nb_elayers = len(dims)
       
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stem = nn.Sequential(
            MinkowskiConvolution(in_channels, dims[0], kernel_size=3, stride=1, dimension=3),
            MinkowskiLayerNorm(dims[0], eps=1e-6),
        ) 

        for i in range(self.nb_elayers):
            encoder_layer = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            cur += depths[i]

            if i < self.nb_elayers - 1:  
                downsample_layer = nn.Sequential(
                    MinkowskiConvolution(dims[i], dims[i+1], kernel_size=3, stride=2, bias=True, dimension=3),
                    MinkowskiLayerNorm(dims[i+1], eps=1e-6),                
                )
                self.downsample_layers.append(downsample_layer)

        """Decoder"""
        last_enc_depth = depths[-1]
        #depths = [1, 1, 1, 1, 1]
        depths = depths[:-1][::-1]
        dims = dims[::-1]
        decoder_embed_dim = 32

        self.nb_dlayers = len(dims) - 1

        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(dp_rates[-last_enc_depth], 0, sum(depths))]
        cur = 0

        for i in range(self.nb_dlayers):
            upsample_layer = nn.Sequential(
                MinkowskiConvolutionTranspose(dims[i] if i==0 else dims[i]+dims[i], dims[i+1], kernel_size=3, stride=2, bias=True, dimension=3),
                MinkowskiLayerNorm(dims[i+1], eps=1e-6),
            )
            self.upsample_layers.append(upsample_layer)

            decoder_layer = nn.Sequential(
                *[Block(dim=dims[i+1] + dims[i+1], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.decoder_layers.append(decoder_layer)
            cur += depths[i]

        """Cls layers"""
        if self.contrastive:
            self.cls_layer = nn.Sequential(
                MinkowskiConvolution(dims[-1], decoder_embed_dim, kernel_size=1, stride=1, dimension=3),
                Block(dim=decoder_embed_dim, drop_path=0., D=3),
                MinkowskiConvolution(decoder_embed_dim, decoder_embed_dim, kernel_size=1, stride=1, dimension=3),
                Block(dim=decoder_embed_dim, drop_path=0., D=3),
                MinkowskiConvolution(decoder_embed_dim, decoder_embed_dim, kernel_size=1, stride=1, dimension=3),
            )
        else:
            self.cls_layers = nn.ModuleList()
            for i in range(self.nb_dlayers):
                cls_layer = MinkowskiConvolution(dims[i+1] + dims[i+1], out_channels, kernel_size=1, stride=1, dimension=3)
                self.cls_layers.append(cls_layer)

        if not self.contrastive:
            """ Pool just for generating downsampled labels """        
            self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3) 

        """ Initialise weights """
        self.apply(_init_weights)

    def forward(self, x, y):
        if self.contrastive:
            ys = None
        else:
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
            x = ME.cat(x, x_enc[i])
            x = self.decoder_layers[i](x)
            if not self.contrastive:
                if i >= (self.nb_dlayers - self.ds_steps):
                    out_cl = self.cls_layers[i](x)
                    out_cls.insert(0, out_cl)

        if self.contrastive:
            out_cls = self.cls_layer(x)
 
        return out_cls, ys

