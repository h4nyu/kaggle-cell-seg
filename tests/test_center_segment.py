from cellseg.center_segment import CenterSegment, CenterCrop
from cellseg.backbones import EfficientNetFPN
import torch


def test_center_segment() -> None:
    image_batch = torch.rand(2, 3, 512, 512)
    backbone = EfficientNetFPN("efficientnet-b0")
    num_classes = 1
    category_feat_range = (4, 6)

    model = CenterSegment(
        num_classes=num_classes,
        backbone=backbone,
        category_feat_range=category_feat_range,
        hidden_channels=64,
        center_crop=CenterCrop(output_size=64),
    )
    model(image_batch)
