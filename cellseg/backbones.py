import torch.nn as nn
from torch import Tensor
from efficientnet_pytorch import EfficientNet
from typing import Protocol, Callable, Any
from .utils import round_to
from .blocks import CSPDarkBlock, ConvBnAct, DarkBlock, DefaultActivation


class FPNLike(Protocol):
    out_channels: list[int]
    strides: list[int]

    def __call__(self, x: Tensor) -> list[Tensor]:
        ...


efficientnet_channels = {
    "efficientnet-b0": [3, 16, 24, 40, 112, 320, 1280],
    "efficientnet-b1": [3, 16, 24, 40, 112, 320, 1280],
    "efficientnet-b2": [3, 16, 24, 48, 120, 352, 1408],
    "efficientnet-b3": [3, 24, 32, 48, 136, 384, 1536],
    "efficientnet-b4": [3, 24, 32, 56, 160, 448, 1792],
    "efficientnet-b5": [3, 24, 40, 64, 176, 512, 2048],
    "efficientnet-b6": [3, 32, 40, 72, 200, 576, 2304],
    "efficientnet-b7": [3, 32, 48, 80, 224, 640, 2560],
}

efficientnet_strides = [1, 2, 4, 8, 16, 32, 32]


class EfficientNetFPN(nn.Module):
    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__()
        self.net = EfficientNet.from_pretrained(name)
        self.out_len = 7
        self.out_channels = efficientnet_channels[name][: self.out_len]
        self.strides = efficientnet_strides

    def forward(self, images: Tensor) -> list[Tensor]:  # P1 - P6, P7 is dropped
        features = self.net.extract_endpoints(images)
        return [images, *features.values()][: self.out_len]


class CSPDarknet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        width_mult: float = 1.0,
        depths: list[int] = [1, 2, 8, 8, 4],
        entry_block_type: Any = DarkBlock,
        act: Callable = DefaultActivation,
    ) -> None:
        super().__init__()
        out_channels = int(width * width_mult)
        self.stem = ConvBnAct(in_channels, round_to(out_channels), 3, 1, act=act)
        stages = []
        block_type = entry_block_type
        self.out_channels = []
        for idx_stage, depth in enumerate(depths):
            block_type = entry_block_type if idx_stage == 0 else CSPDarkBlock
            in_channels = out_channels
            out_channels = min(2 * in_channels, int(width * width_mult * 32))
            block = block_type(
                round_to(in_channels), round_to(out_channels), depth, act=act
            )
            stages.append(block)
            if idx_stage > 1:
                self.out_channels.append(round_to(out_channels))
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        output = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx > 1:
                output.append(x)
        return output
