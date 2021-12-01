import torch
import torch.nn as nn
from torch import Tensor
from fvcore.nn import weight_init
from typing import Callable, Optional
from .blocks import (
    ConvBnAct,
    SpatialAttention,
    CSPSPPBlock,
    DefaultActivation,
    CSPUpBlock,
    CSPPlainBlock,
)
from .coord_conv import CoordConv
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        in_channels: list[int] = [],
        reductions: list[int] = [],
        coord_level: Optional[int] = False,
    ) -> None:
        super().__init__()
        self.coord_conv = CoordConv()
        self.in_convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.coord_level = coord_level
        for idx, in_c in enumerate(in_channels):
            self.in_convs.append(
                nn.Conv2d(
                    in_channels=in_c,
                    out_channels=hidden_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                )
            )

        self.enable_coords = []
        for idx in range(len(reductions) - 1):
            scale_factor = reductions[idx + 1] // reductions[idx]
            if scale_factor == 2:
                self.upsamples.append(
                    nn.ConvTranspose2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=2,
                        stride=2,
                        bias=False,
                    )
                )
            else:
                self.upsamples.append(
                    ConvBnAct(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                    ),
                )
            offset = 0
            if idx == coord_level:
                self.enable_coords.append(True)
                offset = 2
            else:
                self.enable_coords.append(False)

            self.merge_convs.append(
                ConvBnAct(
                    in_channels=hidden_channels + offset,
                    out_channels=hidden_channels,
                ),
            )
        self.coord_conv = CoordConv()
        self.out_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        out = self.in_convs[-1](features[-1])
        for (feat, in_conv, up, merge_conv, enable_coord) in zip(
            features[::-1][1:],
            self.in_convs[::-1][1:],
            self.upsamples[::-1],
            self.merge_convs[::-1],
            self.enable_coords[::-1],
        ):
            out = up(out) + in_conv(feat)
            if enable_coord:
                out = self.coord_conv(out)
            out = merge_conv(out)
        out = self.out_conv(out).sigmoid()
        return out


class CSPUpHead(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        in_channels: list[int] = [],
        reductions: list[int] = [],
        depth: int = 2,
        use_cord: bool = False,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        self.coord_conv = CoordConv()
        self.in_convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.use_cord = use_cord
        self.spp_block = CSPSPPBlock(in_channels[-1], in_channels[-1] // 2, act=act)
        self.up_blocks = nn.ModuleList()  # low-res to high-res
        self.coord_conv = CoordConv()
        coord_offset = 2 if use_cord else 0
        for idx in range(len(in_channels) - 1):
            scale_factor = reductions[-idx - 1] // reductions[-idx - 2]
            if scale_factor == 2:
                self.up_blocks.append(
                    CSPUpBlock(
                        in_channels=(
                            (in_channels[-idx - 1] // 2) + coord_offset,
                            in_channels[-idx - 2],
                        ),
                        out_channels=in_channels[-idx - 2] // 2,
                        depth=depth,
                        act=act,
                    )
                )
            else:
                self.up_blocks.append(
                    CSPPlainBlock(
                        in_channels=(
                            (in_channels[-idx - 1] // 2) + coord_offset,
                            in_channels[-idx - 2],
                        ),
                        out_channels=in_channels[-idx - 2] // 2,
                        depth=depth,
                        act=act,
                    )
                )

        self.out_conv = nn.Conv2d(
            in_channels=in_channels[0] // 2,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )

    def forward(self, inputs: list[Tensor]) -> Tensor:
        h = self.spp_block(inputs[-1])
        for block, x in zip(self.up_blocks, inputs[-2::-1]):
            if self.use_cord:
                h = self.coord_conv(h)
            h = block(h, x)
        h = self.out_conv(h).sigmoid()
        return h


class MaskHead(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int = 1, width: int = 256, depth: int = 4
    ) -> None:
        super().__init__()
        self.in_convs = nn.Sequential(
            *(
                ConvBnAct(
                    in_channels=(in_channels if idx == 0 else width), out_channels=width
                )
                for idx in range(depth)
            )
        )
        self.sam = SpatialAttention(in_channels=width)
        self.deconv = nn.ConvTranspose2d(
            in_channels=width, out_channels=width, kernel_size=2, stride=2, bias=False
        )
        self.act = nn.Mish(inplace=True)
        self.out_conv = nn.Conv2d(width, out_channels, kernel_size=1, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.in_convs.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                weight_init.c2_msra_fill(m)
        weight_init.c2_msra_fill(self.deconv)
        nn.init.normal_(self.out_conv.weight, std=0.001)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_convs(x)
        h = self.sam(h)
        h = self.deconv(h)
        h = self.act(h)
        return self.out_conv(h)
