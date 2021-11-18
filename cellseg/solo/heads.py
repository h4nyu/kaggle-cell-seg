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
        self.convs_all_levels = nn.ModuleList()
        self.convs_all_levels.append(
            CovNormAct(
                in_channels=channels[0],
                out_channels=hidden_channels,
                activation=None,  # TODO: why version is different
            ),
        )
        down_counts = torch.log2(torch.tensor(reductions)).long()
        down_counts = down_counts - down_counts[0]
        for level_idx, (in_channels, down_count) in enumerate(
            zip(channels[1:], down_counts[1:]), 1
        ):
            convs_per_level = nn.Sequential()
            convs_per_level.add_module(
                f"conv{level_idx}",
                CovNormAct(
                    in_channels=in_channels + 2,
                    out_channels=hidden_channels,
                ),
            )

            for j in range(down_count):
                if j != 0:
                    convs_per_level.add_module(
                        f"conv{j}",
                        CovNormAct(
                            in_channels=hidden_channels,
                            out_channels=hidden_channels,
                        ),
                    )
                upsample = nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                )
                convs_per_level.add_module(f"upsample{j}", upsample)
            self.convs_all_levels.append(convs_per_level)

        self.coord_conv = CoordConv()
        self.out_conv = CovNormAct(
            in_channels=hidden_channels,
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
        out = torch.sigmoid(out)
        return out
