import pytest
import torch
from cellseg.yolox import MaskYolo, ToBoxes
from cellseg.backbones import EfficientNetFPN
from cellseg.necks import CSPNeck

@pytest.fixture
def mask_yolo() -> MaskYolo:
    backbone = EfficientNetFPN("efficientnet-b0")
    num_classes = 2
    mask_size = 16
    top_fpn_level = 4
    neck = CSPNeck(
        in_channels=backbone.out_channels[:top_fpn_level],
        out_channels=backbone.out_channels[:top_fpn_level],
        reductions=backbone.reductions[:top_fpn_level],
    )
    to_boxes = ToBoxes()
    return MaskYolo(
        backbone=backbone,
        neck=neck,
        mask_size=mask_size,
        to_boxes=to_boxes,
        num_classes=num_classes,
        top_fpn_level=top_fpn_level,
    )

def test_mask_yolo(mask_yolo:MaskYolo) -> None:
    image_size = 256
    images = torch.rand(2, 3, image_size, image_size)
    masks = mask_yolo(images)

def test_mask_yolo_mask_branch(mask_yolo:MaskYolo) -> None:
    top_fpn_level = mask_yolo.top_fpn_level
    out_channels = mask_yolo.backbone.out_channels[:top_fpn_level]
    reductions = mask_yolo.backbone.reductions[:top_fpn_level]
    image_size = 128
    features = [
        torch.rand(1, c, image_size // r, image_size // r)
        for (c, r) in zip(out_channels, reductions)
    ]
    boxes = torch.tensor([
        [ 10, 20, 30, 40],
        [ 10, 20, 30, 40],
    ]).float()
    box_batch = [boxes]
    masks = mask_yolo.mask_branch(box_batch, features)
