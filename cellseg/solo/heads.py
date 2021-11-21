import torch
import torch.nn as nn
from torch import Tensor
from .convs import CovNormAct
from .coord_conv import CoordConv
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_classes: int,
        channels: list[int] = [],
        reductions: list[int] = [],
    ) -> None:
        super().__init__()
        self.coord_conv = CoordConv()
        self.in_convs = nn.ModuleList()
        self.merge_convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for idx, in_channels in enumerate(channels):
            self.in_convs.append(
                CovNormAct(
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
                nn.Sequential(
                    CoordConv(),
                    CovNormAct(
                        in_channels=hidden_channels + 2,
                        out_channels=hidden_channels,
                    )
                )
            )

        self.out_conv = CovNormAct(
            in_channels=hidden_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            activation=torch.sigmoid,
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        out = self.in_convs[-1](features[-1])
        for feat, in_conv, up, merge_conv in zip(
            features[::-1][1:],
            self.in_convs[::-1][1:],
            self.upsamples[::-1],
            self.merge_convs[::-1],
        ):
            out = merge_conv(up(out) + in_conv(feat))
        out = self.out_conv(out)
        return out
