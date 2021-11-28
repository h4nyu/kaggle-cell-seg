import torch.nn as nn
from torch import Tensor
from efficientnet_pytorch import EfficientNet
from typing import Protocol


class FPNLike(Protocol):
    channels: list[int]
    reductions: list[int]

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

efficientnet_reductions = [1, 2, 4, 8, 16, 32, 32]


class EfficientNetFPN(nn.Module):
    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__()
        self.net = EfficientNet.from_pretrained(name)
        self.out_len = 7
        self.channels = efficientnet_channels[name][: self.out_len]
        self.reductions = efficientnet_reductions

    def forward(self, images: Tensor) -> list[Tensor]:  # P1 - P6, P7 is dropped
        features = self.net.extract_endpoints(images)
        return [images, *features.values()][: self.out_len]
