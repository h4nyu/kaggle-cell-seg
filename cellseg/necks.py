import torch
import torch.nn as nn
from typing import Callable, Protocol
from torch import Tensor
from .blocks import (
    DefaultActivation,
    ConvBnAct,
    CSPSPPBlock,
    CSPUpBlock,
    CSPDownBlock,
    CSPPlainBlock,
)


class NeckLike(Protocol):
    out_channels: list[int]
    reductions: list[int]

    def __call__(self, x: list[Tensor]) -> list[Tensor]:
        ...


class CSPNeck(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        reductions: list[int],
        depth: int = 2,
        act: Callable[[Tensor], Tensor] = DefaultActivation,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.reductions = reductions
        self.spp_block = CSPSPPBlock(in_channels[-1], in_channels[-1] // 2, act=act)

        self.up_blocks = nn.ModuleList()  # low-res to high-res
        for idx in range(len(in_channels) - 1):
            scale_factor = reductions[-idx - 1] // reductions[-idx - 2]
            if scale_factor == 2:
                self.up_blocks.append(
                    CSPUpBlock(
                        in_channels=(
                            in_channels[-idx - 1] // 2,
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
                            in_channels[-idx - 1] // 2,
                            in_channels[-idx - 2],
                        ),
                        out_channels=in_channels[-idx - 2] // 2,
                        depth=depth,
                        act=act,
                    )
                )

        self.down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            scale_factor = reductions[idx + 1] // reductions[idx]
            if scale_factor == 2:
                self.down_blocks.append(
                    CSPDownBlock(
                        in_channels=(
                            in_channels[idx] // 2,
                            in_channels[idx + 1] // 2,
                        ),
                        out_channels=in_channels[idx + 1] // 2,
                        depth=depth,
                        act=act,
                    )
                )
            else:
                self.down_blocks.append(
                    CSPPlainBlock(
                        in_channels=(
                            in_channels[idx] // 2,
                            in_channels[idx + 1] // 2,
                        ),
                        out_channels=in_channels[idx + 1] // 2,
                        depth=depth,
                        act=act,
                    )
                )

        self.out_convs = nn.ModuleList(
            [
                ConvBnAct(ic // 2, oc, 3, act=act)
                for ic, oc in zip(in_channels, out_channels)
            ]
        )

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        # assume that inputs are orderd high-res to low-res.
        hs1 = []
        h = self.spp_block(inputs[-1])
        hs1.append(h)
        for block, x in zip(self.up_blocks, inputs[-2::-1]):
            h = block(h, x)
            hs1.append(h)
        hs2 = []
        h = hs1[-1]
        hs2.append(h)
        for block, x in zip(self.down_blocks, hs1[-2::-1]):
            h = block(h, x)
            hs2.append(h)
        return [m(x) for m, x in zip(self.out_convs, hs2)]
