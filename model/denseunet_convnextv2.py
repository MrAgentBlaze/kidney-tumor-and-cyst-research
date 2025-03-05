import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import (
    LayerNorm,
    GRN,
)


# Custom weight initialization function
def _init_weights(m):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)    
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, kernel_size=5, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input + self.drop_path(x)
        return x


class DenseUNetConvNeXtV2(nn.Module):

    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)
        
        self.contrastive = args.contrastive
        self.finetuning = args.finetuning
        self.ds_steps = args.ds_steps

        """Encoder"""
        depths=[2, 4, 4, 8, 8, 8]
        dims = (16, 32, 64, 128, 256, 512)
        kernel_size = 3
        drop_path_rate = 0.0

        assert len(depths) == len(dims)

        self.nb_elayers = len(dims)
       
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, dims[0], kernel_size=1, stride=1, bias=False),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        ) 

        for i in range(self.nb_elayers):
            encoder_layer = nn.Sequential(
                *[Block(dim=dims[i], kernel_size=kernel_size, drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            cur += depths[i]

            if i < self.nb_elayers - 1:  
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),                
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True),
                )
                self.downsample_layers.append(downsample_layer)

        """Decoder"""
        last_enc_depth = depths[-1]
        depths = [2, 2, 2, 2, 2]
        dims = dims[::-1]
        decoder_embed_dim = 32

        self.nb_dlayers = len(dims) - 1

        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(dp_rates[-last_enc_depth], 0, sum(depths))]
        cur = 0

        for i in range(self.nb_dlayers):
            upsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"), 
                nn.ConvTranspose3d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True),
            )
            self.upsample_layers.append(upsample_layer)

            decoder_layer = nn.Sequential(
                *[Block(dim=dims[i+1], kernel_size=kernel_size, drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.decoder_layers.append(decoder_layer)
            cur += depths[i]

        """Cls layers"""
        if self.contrastive:
            self.cls_layer = nn.Sequential(
                nn.Conv3d(dims[-1], decoder_embed_dim, kernel_size=1, stride=1, bias=False),
                Block(dim=decoder_embed_dim, kernel_size=kernel_size, drop_path=0.),
                nn.Conv3d(decoder_embed_dim, decoder_embed_dim, kernel_size=1, stride=1, bias=False),
                Block(dim=decoder_embed_dim, kernel_size=kernel_size, drop_path=0.),
                nn.Conv3d(decoder_embed_dim, decoder_embed_dim, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.cls_layers = nn.ModuleList()
            for i in range(self.nb_dlayers):
                cls_layer = nn.Sequential(
                    LayerNorm(dims[i+1], eps=1e-6, data_format="channels_first"),
                    Block(dim=dims[i+1], kernel_size=kernel_size, drop_path=0.),
                    nn.Conv3d(dims[i+1], out_channels, kernel_size=1, stride=1, bias=False),
                )
                self.cls_layers.append(cls_layer)

        if not self.contrastive:
            """ Pool just for generating downsampled labels """        
            self.pool = nn.AvgPool3d(kernel_size=2, stride=2) 

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
            _, _, d, h, w = x.shape
            x = x + x_enc[i][:, :, :d, :h, :w]
            x = self.decoder_layers[i](x)
            if not self.contrastive:
                if i >= (self.nb_dlayers - self.ds_steps):
                    out_cl = self.cls_layers[i](x)
                    out_cls.insert(0, out_cl)

        if self.contrastive:
            out_cls = self.cls_layer(x)
 
        return out_cls, ys

