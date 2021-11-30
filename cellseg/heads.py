import torch
import torch.nn as nn
from torch import Tensor
from fvcore.nn import weight_init
from .blocks import ConvBnAct, SpatialAttention
from .coord_conv import CoordConv
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        channels: list[int] = [],
        reductions: list[int] = [],
        use_cord: bool = False,
    ) -> None:
        super().__init__()
        self.coord_conv = CoordConv()
        self.in_convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.use_cord = use_cord
        for idx, in_channels in enumerate(channels):
            self.in_convs.append(
                ConvBnAct(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                )
            )

        for idx in range(len(reductions) - 1):
            scale_factor = reductions[idx + 1] // reductions[idx]
            self.upsamples.append(
                nn.Upsample(
                    scale_factor=scale_factor, mode="bilinear", align_corners=False
                )
            )

            self.merge_convs.append(
                ConvBnAct(
                    in_channels=hidden_channels + 2
                    if self.use_cord
                    else hidden_channels,
                    out_channels=hidden_channels,
                ),
            )
        self.coord_conv = CoordConv()
        self.out_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            bias=False,
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        out = self.in_convs[-1](features[-1])
        for feat, in_conv, up, merge_conv in zip(
            features[::-1][1:],
            self.in_convs[::-1][1:],
            self.upsamples[::-1],
            self.merge_convs[::-1],
        ):
            if self.use_cord:
                out = merge_conv(self.coord_conv(up(out) + in_conv(feat)))
            else:
                out = merge_conv(up(out) + in_conv(feat))
        out = self.out_conv(out).sigmoid()
        return out


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
