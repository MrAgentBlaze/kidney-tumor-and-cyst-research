import torch
import torch.nn as nn
import MinkowskiEngine as ME
from timm.models.layers import trunc_normal_
from .utils import MinkowskiLayerNorm, MinkowskiGRN, MinkowskiDropPath

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiConvolutionTranspose,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
)


def _init_weights(m):
    """Initialise weights using truncated normal distribution."""
    if isinstance(m, MinkowskiConvolution):
        trunc_normal_(m.kernel, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiConvolutionTranspose):
        trunc_normal_(m.kernel, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiDepthwiseConvolution):
        trunc_normal_(m.kernel, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, MinkowskiLinear):
        trunc_normal_(m.linear.weight, std=0.02)
        if m.linear.bias is not None:
            nn.init.constant_(m.linear.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class Block(nn.Module):
    """Sparse ConvNeXtV2 block.

    Args:
        dim: Number of input channels.
        kernel_size: Convolution kernel size.
        drop_path: Stochastic depth rate.
        D: Spatial dimensionality.
    """

    def __init__(self, dim, kernel_size=3, drop_path=0.0, D=3):
        super().__init__()
        self.dwconv = MinkowskiDepthwiseConvolution(
            dim, kernel_size=kernel_size, bias=True, dimension=D
        )
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)
        self.act = MinkowskiGELU()
        self.grn = MinkowskiGRN(4 * dim)
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.drop_path = MinkowskiDropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = residual + self.drop_path(x)
        return x


class MinkUNetConvNeXtV2(nn.Module):
    """Sparse 3D U-Net with ConvNeXtV2 blocks, built on MinkowskiEngine.

    The encoder has six hierarchical stages with increasing feature sizes
    (16, 32, 64, 128, 256, 512) and depths (2, 4, 4, 8, 8, 8). The decoder
    mirrors the encoder with five stages of two blocks each. Deep supervision
    is applied at multiple decoder stages.
    """

    def __init__(self, in_channels, out_channels, D=3, args=None):
        super().__init__()
        self.ds_steps = args.ds_steps

        # Encoder
        depths = [2, 4, 4, 8, 8, 8]
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
            MinkowskiConvolution(in_channels, dims[0], kernel_size=1, stride=1, dimension=D),
            MinkowskiLayerNorm(dims[0], eps=1e-6),
        )

        for i in range(self.nb_elayers):
            encoder_layer = nn.Sequential(
                *[Block(dim=dims[i], kernel_size=kernel_size,
                        drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.encoder_layers.append(encoder_layer)
            cur += depths[i]

            if i < self.nb_elayers - 1:
                downsample_layer = nn.Sequential(
                    MinkowskiLayerNorm(dims[i], eps=1e-6),
                    MinkowskiConvolution(dims[i], dims[i + 1], kernel_size=2,
                                        stride=2, bias=True, dimension=D),
                )
                self.downsample_layers.append(downsample_layer)

        # Decoder
        last_enc_depth = depths[-1]
        depths = [2, 2, 2, 2, 2]
        dims = dims[::-1]

        self.nb_dlayers = len(dims) - 1
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(
            dp_rates[-last_enc_depth], 0, sum(depths))]
        cur = 0

        for i in range(self.nb_dlayers):
            upsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolutionTranspose(dims[i], dims[i + 1], kernel_size=2,
                                              stride=2, bias=True, dimension=D),
            )
            self.upsample_layers.append(upsample_layer)

            decoder_layer = nn.Sequential(
                *[Block(dim=dims[i + 1], kernel_size=kernel_size,
                        drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.decoder_layers.append(decoder_layer)
            cur += depths[i]

        # Classification heads for deep supervision
        self.cls_layers = nn.ModuleList()
        for i in range(self.nb_dlayers):
            cls_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i + 1], eps=1e-6),
                Block(dim=dims[i + 1], kernel_size=kernel_size, drop_path=0.0, D=D),
                MinkowskiConvolution(dims[i + 1], out_channels, kernel_size=1,
                                    stride=1, dimension=D),
            )
            self.cls_layers.append(cls_layer)

        # Pooling for generating downsampled labels
        self.pool = ME.MinkowskiAvgPooling(kernel_size=2, stride=2, dimension=3)

        self.apply(_init_weights)

    def forward(self, x, y):
        # Generate downsampled labels for deep supervision
        ys = []
        for i in range(self.ds_steps):
            y_aux = y.detach() if i == 0 else self.pool(y_aux)
            ys.append(y_aux)

        # Encoder
        x = self.stem(x)
        x_enc = []
        for i in range(self.nb_elayers):
            x = self.encoder_layers[i](x)
            if i < self.nb_elayers - 1:
                x_enc.append(x)
                x = self.downsample_layers[i](x)
        x_enc = x_enc[::-1]

        # Decoder with skip connections (element-wise summation)
        out_cls = []
        for i in range(self.nb_dlayers):
            x = self.upsample_layers[i](x)
            x = x + x_enc[i]
            x = self.decoder_layers[i](x)
            if i >= (self.nb_dlayers - self.ds_steps):
                out_cl = self.cls_layers[i](x)
                out_cls.insert(0, out_cl)

        return out_cls, ys

