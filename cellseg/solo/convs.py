import torch.nn as nn
import torch


class CovNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.norm = nn.BatchNorm2d(
            num_features=out_channels,
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.conv(feat)
        out = self.norm(out)
        out = self.act(out)
        return out
