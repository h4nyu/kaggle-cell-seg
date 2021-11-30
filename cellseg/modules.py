import torch
import torch.nn as nn
from torch import Tensor
from fvcore import weight_init
from .convs import CovNormAct


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = torch.cat(
            [x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)], dim=1
        )  # [b, 2, 1, 1]
        return x * self.conv(h).sigmoid()

    def _init_weights(self) -> None:
        weight_init.c2_msra_fill(self.conv)


class MaskHead(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int = 1, width: int = 256, depth: int = 4
    ) -> None:
        super().__init__()
        self.in_convs = nn.Sequential(
            *(
                CovNormAct(
                    in_channels=(in_channels if idx == 0 else width), out_channels=width
                )
                for idx in range(depth)
            )
        )
        self.sam = SpatialAttention(width)
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
        # nn.init.normal_(self.out_conv.weight, mean=0.06, std=1.0)
        nn.init.normal_(self.out_conv.weight, std=0.001)
        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        h = self.in_convs(x)
        h = self.sam(h)
        h = self.deconv(h)
        h = self.act(h)
        return self.out_conv(h)
