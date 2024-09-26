import math
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
        nn.init.constant_(m.bias, 0)
    if isinstance(m, ME.MinkowskiDepthwiseConvolution):
        nn.init.trunc_normal_(m.kernel, std=.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, ME.MinkowskiLinear):
        nn.init.trunc_normal_(m.linear.weight, std=.02)
        nn.init.constant_(m.linear.bias, 0)
    if isinstance(m, nn.Conv2d):
        w = m.weight.data
        nn.init.trunc_normal_(w.view([w.shape[0], -1]))
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MinkUNetConvNeXtV2(nn.Module):
    CHANNELS = (8, 16, 32, 64, 128, 256, 512)
    #CHANNELS = (4, 8, 16, 32, 64, 128, 256)

    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)
        
        ch = self.CHANNELS
        #self.ds_steps = args.ds_steps

        """Encoder"""
        depths=[3, 3, 9, 3]
        dims=[96, 192, 384, 768]     
        #depths=[2, 2, 6, 2]
        #dims=(32, 64, 128, 256)
        drop_path_rate=0.

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            MinkowskiConvolution(in_channels, dims[0], kernel_size=4, stride=4, dimension=3),
            MinkowskiLayerNorm(dims[0], eps=1e-6),
        ) 
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=3)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages_enc = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.stages_enc.append(stage)
            cur += depths[i]
            
        # Decoder projection
        decoder_embed_dim = 512
        patch_size = 32
        num_classes = 3
        self.proj = nn.Conv3d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1)

        # Segmentation Head to output patch-level logits for each class
        self.segmentation_head = nn.Conv3d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 3 * num_classes,  # Patchified output
            kernel_size=1)

        self.patch_size = patch_size  # Store patch size for unpatchify

        """ Initialise weights """
        self.apply(_init_weights)


    def patchify(self, imgs):
        """
        imgs: (N, num_classes, H, W, D)
        x: (N, L, patch_size**3 * num_classes)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0, \
            "Image dimensions must be divisible by the patch size"
        
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        d = imgs.shape[4] // p
        
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p, d, p))
        x = torch.einsum('nchpwqds->nhwdpqsc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3 * imgs.shape[1]))  # Combine patches into sequence
    
        return x
        
    def unpatchify(self, x, H, W, D):
        """
        x: (N, d*h*w, patch_size**3 * num_classes)
        D, H, W: Original dimensions of the input 3D image
        Returns:
            imgs: (N, num_classes, D, H, W)
        """
        p = self.patch_size

        h = math.ceil(H / p)
        w = math.ceil(W / p)
        d = math.ceil(D / p)
        
        assert h * w * d == x.shape[1], "Patch sequence length does not match the expected grid size"
    
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, -1))  # Reshape back to 3D patches
        x = torch.einsum('nhwdpqsc->nchpwqds', x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, w * p, d * p))  # Reconstruct full image volume        
        imgs = imgs[:, :, :H, :W, :D]
        
        return imgs
  
    def forward(self, x, y):        
        # encoder
        x = self.downsample_layers[0](x)
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages_enc[i](x)
            
        # densify
        x = x.dense()[0]
        y = y.dense()[0]

        # project and get patch-level segmentation logits
        x = self.proj(x)
        x = self.segmentation_head(x)
        
        # Reshape the output
        n, c, d, h, w = x.shape
        x = x.reshape(n, c, -1)  # Flatten spatial dims into patch sequence
        x = torch.einsum('ncl->nlc', x)
        
        return x, y

