import torch.nn as nn
from torch import Tensor
from .convs import CovNormAct
from .coord_conv import CoordConv


class MaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_classes: int,
        fpn_length: int,
    ) -> None:
        super().__init__()
        self.convs_all_levels = nn.ModuleList()
        self.convs_all_levels.append(
            CovNormAct(
                in_channels,
                out_channels,
            ),
        )

        for level_idx in range(1, fpn_length):
            convs_per_level = nn.Sequential()
            convs_per_level.add_module(
                f"conv{level_idx}",
                CovNormAct(
                    in_channels + 2,
                    out_channels,
                ),
            )
            convs_per_level.add_module(
                f"upsample{level_idx}",
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )
            for j in range(1, level_idx):
                convs_per_level.add_module(
                    f"conv{j}",
                    CovNormAct(
                        out_channels,
                        out_channels,
                    ),
                )
                upsample = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                )
                convs_per_level.add_module("upsample" + str(j), upsample)
            self.convs_all_levels.append(convs_per_level)

        self.coord_conv = CoordConv()
        self.out_conv = CovNormAct(
            in_channels=out_channels,
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        conved_sum = self.convs_all_levels[0](features[0])
        for feat, conv in zip(features[1:], self.convs_all_levels[1:]):
            coord_feat = self.coord_conv(feat)
            conved = conv(coord_feat)
            conved_sum += conved
        out = self.out_conv(conved_sum)
        return out

