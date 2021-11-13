import torch
from cellseg.solo import Solo
from cellseg.backbones import EfficientNetFPN


def test_solo() -> None:
    image_batch = torch.rand(1, 3, 512, 512)
    backbone = EfficientNetFPN("efficientnet-b1")
    solo = Solo(backbone=backbone, in_channels=128, out_channels=64, grid_size=64)
    solo(image_batch)
