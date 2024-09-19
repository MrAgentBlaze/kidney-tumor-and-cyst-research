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


class MinkUNetConvNeXtV2(nn.Module):
    CHANNELS = (8, 16, 32, 64, 128, 256, 512)
    #CHANNELS = (4, 8, 16, 32, 64, 128, 256)

    def __init__(self, in_channels, out_channels, D=3, args=None):
        nn.Module.__init__(self)

        ch = self.CHANNELS
        self.ds_steps = args.ds_steps

        """Encoder"""
        
        # Block 1
        self.eblock1 = nn.Sequential(
            Block(dim=in_channels, drop_path=0, D=3),
            MinkowskiLayerNorm(1, eps=1e-6),
            ME.MinkowskiConvolution(1, ch[0], kernel_size=3, stride=2, bias=True, dimension=3),
            Block(dim=ch[0], drop_path=0, D=3),
        )

        # Block 2
        self.eblock2 = nn.Sequential(
            MinkowskiLayerNorm(ch[0], eps=1e-6),
            ME.MinkowskiConvolution(ch[0], ch[1], kernel_size=3, stride=2, bias=True, dimension=3),
            Block(dim=ch[1], drop_path=0, D=3),
        )

        # Block 3
        self.eblock3 = nn.Sequential(
            MinkowskiLayerNorm(ch[1], eps=1e-6),
            ME.MinkowskiConvolution(ch[1], ch[2], kernel_size=3, stride=2, bias=True, dimension=3),
            Block(dim=ch[2], drop_path=0, D=3),
        )

        # Block 4
        self.eblock4 = nn.Sequential(
            MinkowskiLayerNorm(ch[2], eps=1e-6),
            ME.MinkowskiConvolution(ch[2], ch[3], kernel_size=3, stride=2, bias=True, dimension=3),
            Block(dim=ch[3], drop_path=0, D=3),
        )

        # Block 5
        self.eblock5 = nn.Sequential(
            MinkowskiLayerNorm(ch[3], eps=1e-6),
            ME.MinkowskiConvolution(ch[3], ch[4], kernel_size=3, stride=2, bias=True, dimension=3),
            Block(dim=ch[4], drop_path=0, D=3),
        )

        # Block 6
        self.eblock6 = nn.Sequential(
            MinkowskiLayerNorm(ch[4], eps=1e-6),
            ME.MinkowskiConvolution(ch[4], ch[5], kernel_size=3, stride=2, bias=True, dimension=3),
            Block(dim=ch[5], drop_path=0, D=3),
        )

        # Block 7
        self.eblock7 = nn.Sequential(
            MinkowskiLayerNorm(ch[5], eps=1e-6),
            ME.MinkowskiConvolution(ch[5], ch[6], kernel_size=3, stride=2, bias=True, dimension=3),
            Block(dim=ch[6], drop_path=0, D=3),
        )
        
        """Decoder"""
        
        # Block 1
        self.dblock1 = nn.Sequential(
            Block(dim=ch[6], drop_path=0, D=3),
            MinkowskiLayerNorm(ch[6], eps=1e-6),
            ME.MinkowskiConvolutionTranspose(
                ch[6], ch[6], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            Block(dim=ch[6], drop_path=0, D=3),
            MinkowskiLayerNorm(ch[6], eps=1e-6),
        )
        # Final 1
        if self.ds_steps > 6:
            self.dblock1_cls = nn.Sequential(
                ME.MinkowskiConvolution(
                    ch[6], out_channels, kernel_size=1, bias=True, dimension=3
                ),
                Block(dim=out_channels, drop_path=0, D=3),
            )

        # Block 2
        self.dblock2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[6]+ch[5], ch[5], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            Block(dim=ch[5], drop_path=0, D=3),
            MinkowskiLayerNorm(ch[5], eps=1e-6),
        )
        # Final 2
        if self.ds_steps > 5:
            self.dblock2_cls = nn.Sequential(
                ME.MinkowskiConvolution(
                    ch[5], out_channels, kernel_size=1, bias=True, dimension=3
                ),
                Block(dim=out_channels, drop_path=0, D=3),
            )

       
        # Block 3
        self.dblock3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[5]+ch[4], ch[4], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            Block(dim=ch[4], drop_path=0, D=3),
            MinkowskiLayerNorm(ch[4], eps=1e-6),
        )
        # Final 3
        if self.ds_steps > 4:
            self.dblock3_cls = nn.Sequential(
                ME.MinkowskiConvolution(
                    ch[4], out_channels, kernel_size=1, bias=True, dimension=3
                ),
                Block(dim=out_channels, drop_path=0, D=3),
            )

        # Block 4
        self.dblock4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[4]+ch[3], ch[3], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            Block(dim=ch[3], drop_path=0, D=3),
            MinkowskiLayerNorm(ch[3], eps=1e-6),
        )
        # Final 4
        if self.ds_steps > 3:
            self.dblock4_cls = nn.Sequential(
                ME.MinkowskiConvolution(
                    ch[3], out_channels, kernel_size=1, bias=True, dimension=3
                ),
                Block(dim=out_channels, drop_path=0, D=3),
            )

        # Block 5
        self.dblock5 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[3]+ch[2], ch[2], kernel_size=2, stride=2, bias=True, dimension=3
            ),   
            Block(dim=ch[2], drop_path=0, D=3),
            MinkowskiLayerNorm(ch[2], eps=1e-6),
        )
        # Final 5
        if self.ds_steps > 2:
            self.dblock5_cls = nn.Sequential(
                ME.MinkowskiConvolution(
                    ch[2], out_channels, kernel_size=1, bias=True, dimension=3
                ),
                Block(dim=out_channels, drop_path=0, D=3),
            )

        # Block 6
        self.dblock6 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[2]+ch[1], ch[1], kernel_size=2, stride=2, bias=True, dimension=3
            ),    
            Block(dim=ch[1], drop_path=0, D=3),
            MinkowskiLayerNorm(ch[1], eps=1e-6),
        )
        # Final 6
        if self.ds_steps > 1:
            self.dblock6_cls = nn.Sequential(
                ME.MinkowskiConvolution(
                    ch[1], out_channels, kernel_size=1, bias=True, dimension=3
                ),
                Block(dim=out_channels, drop_path=0, D=3),
            )

        # Block 7
        self.dblock7 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                ch[1]+ch[0], ch[0], kernel_size=2, stride=2, bias=True, dimension=3
            ),
            Block(dim=ch[0], drop_path=0, D=3),
            MinkowskiLayerNorm(ch[0], eps=1e-6),
        )
        # Final 7
        self.dblock7_cls = nn.Sequential(
            ME.MinkowskiConvolution(
                ch[0], out_channels, kernel_size=1, bias=True, dimension=3
            ),
            Block(dim=out_channels, drop_path=0, D=3),
        )

        """ Max pool just for generating downsampled labels """        
        self.max_pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=3)


    def forward(self, x, y):
        """ Generate labels for deep supervision """
        ys = [y.detach()]
        for i in range(1, self.ds_steps):
            y = self.max_pool(y)
            ys.append(y.detach())

        """ Encoder """
        out_e1 = self.eblock1(x)  # tensor_stride == [2, 2, 2]
        out_e2 = self.eblock2(out_e1)  # tensor_stride == [4, 4, 4]
        out_e3 = self.eblock3(out_e2)  # tensor_stride == [8, 8, 8]
        out_e4 = self.eblock4(out_e3)  # tensor_stride == [16, 16, 16]
        out_e5 = self.eblock5(out_e4)  # tensor_stride == [32, 32, 32]
        out_e6 = self.eblock6(out_e5)  # tensor_stride == [64, 64, 64]
        out_e7 = self.eblock7(out_e6)  # tensor_stride == [128, 128, 128]

        """ Decoder """ 
        out_cls = []

        out_d1 = self.dblock1(out_e7)  # tensor_stride == [64, 64, 64]
        if self.ds_steps > 6:
            out_cl = self.dblock1_cls(out_d1)
            out_cls.insert(0, out_cl)

        out_d1e6 = ME.cat(out_d1, out_e6)
        out_d2 = self.dblock2(out_d1e6)  # tensor_stride == [32, 32, 32]
        if self.ds_steps > 5:
            out_cl = self.dblock2_cls(out_d2)
            out_cls.insert(0, out_cl)

        out_d2e5 = ME.cat(out_d2, out_e5)
        out_d3 = self.dblock3(out_d2e5)  # tensor_stride == [16, 16, 16]
        if self.ds_steps > 4:
            out_cl = self.dblock3_cls(out_d3)
            out_cls.insert(0, out_cl)

        out_d3e4 = ME.cat(out_d3, out_e4)
        out_d4 = self.dblock4(out_d3e4)  # tensor_stride == [8, 8, 8]
        if self.ds_steps > 3:
            out_cl = self.dblock4_cls(out_d4)
            out_cls.insert(0, out_cl)

        out_d4e3 = ME.cat(out_d4, out_e3)
        out_d5 = self.dblock5(out_d4e3)  # tensor_stride == [4, 4, 4]
        if self.ds_steps > 2:
            out_cl = self.dblock5_cls(out_d5)
            out_cls.insert(0, out_cl)

        out_d5e2 = ME.cat(out_d5, out_e2)
        out_d6 = self.dblock6(out_d5e2)  # tensor_stride == [2, 2, 2]
        if self.ds_steps > 1:
            out_cl = self.dblock6_cls(out_d6)
            out_cls.insert(0, out_cl)

        out_d6e1 = ME.cat(out_d6, out_e1)
        out_d7 = self.dblock7(out_d6e1)  # tensor_stride == [1, 1, 1]
        out_cl = self.dblock7_cls(out_d7)
        out_cls.insert(0, out_cl)
        
        return out_cls, ys

