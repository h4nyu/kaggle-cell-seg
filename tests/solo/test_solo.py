import torch
from cellseg.solo import Solo
from cellseg.backbones import EfficientNetFPN


def test_mask_head() -> None:
    image_batch = torch.rand(1, 3, 512, 512)
    backbone = EfficientNetFPN()
    solo = Solo(backbone=backbone, in_channels=128, out_channels=64, grid_size=64)
    solo(image_batch)
