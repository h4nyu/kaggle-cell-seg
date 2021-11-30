import torch
from torch import Tensor
from typing import Callable, Any
import torch.nn as nn
from .blocks import ConvBnAct, DefaultActivation
from cellseg.utils import round_to
import math


class DecoupledHeadUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width_mult: float = 1.0,
        num_classes: int = 80,
        act: Callable[[Tensor], Tensor] = DefaultActivation,
    ):
        super().__init__()
        hidden_channels = round_to(256 * width_mult)
        self.stem_conv = ConvBnAct(
            in_channels=in_channels, out_channels=hidden_channels, kernel_size=1
        )
        self.reg_conv = nn.Sequential(
            ConvBnAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
            ConvBnAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
        )
        self.reg_out_conv = nn.Conv2d(
            in_channels=hidden_channels, out_channels=4, kernel_size=1
        )
        self.obj_out_conv = nn.Conv2d(
            in_channels=hidden_channels, out_channels=1, kernel_size=1
        )
        self.cls_conv = nn.Sequential(
            ConvBnAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
            ConvBnAct(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                act=act,
            ),
        )
        self.cls_out_conv = nn.Conv2d(hidden_channels, num_classes, 1)
        self._init_weights()

    def _init_weights(self, prior_prob: Any = 1e-2) -> None:
        for m in [self.cls_out_conv, self.obj_out_conv]:
            nn.init.constant_(m.bias, -math.log((1 - prior_prob) / prior_prob))  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem_conv(x)
        h1 = self.reg_conv(h)
        y_reg = self.reg_out_conv(h1)
        y_obj = self.obj_out_conv(h1)
        h2 = self.cls_conv(h)
        y_cls = self.cls_out_conv(h2)
        return torch.cat([y_reg, y_obj, y_cls], dim=1)


class YoloxHead(nn.Module):
    def __init__(
        self,
        in_channels: list[int] = [256, 512, 1024],
        num_classes: int = 80,
        width_mult: float = 1.0,
        act: Callable[[Tensor], Tensor] = nn.Mish(inplace=True),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.heads = nn.ModuleList(
            [
                DecoupledHeadUnit(
                    in_channels=in_chs,
                    num_classes=num_classes,
                    width_mult=width_mult,
                    act=act,
                )
                for in_chs in in_channels
            ]
        )

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        return [
            m(x) for m, x in zip(self.heads, inputs)
        ]  # list of [b, num_classes + 5, h, w]
