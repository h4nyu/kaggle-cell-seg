import pytest
import torch
from torch import Tensor
from cellseg.yolox import MaskYolo, Criterion
from cellseg.backbones import EfficientNetFPN
from cellseg.necks import CSPNeck


@pytest.fixture
def mask_yolo() -> MaskYolo:
    backbone = EfficientNetFPN("efficientnet-b0")
    num_classes = 2
    mask_size = 16
    box_feat_range = (2, 5)
    mask_feat_range = (0, 3)
    neck = CSPNeck(
        in_channels=backbone.out_channels,
        out_channels=backbone.out_channels,
        reductions=backbone.reductions,
    )
    return MaskYolo(
        backbone=backbone,
        neck=neck,
        mask_size=mask_size,
        num_classes=num_classes,
        box_feat_range=box_feat_range,
        mask_feat_range=mask_feat_range,
    )


@pytest.fixture
def targets() -> tuple[list[Tensor], list[Tensor]]:
    masks0 = torch.zeros(1, 128, 128, dtype=torch.bool)
    masks0[0, 10:20, 30:40] = True
    masks1 = torch.zeros(2, 128, 128, dtype=torch.bool)
    masks1[0, 10:20, 30:40] = True
    masks1[1, 20:30, 30:40] = True
    mask_batch = [masks0, masks1]
    label_batch = [torch.zeros(len(m)).long() for m in mask_batch]
    return mask_batch, label_batch


def test_mask_yolo(mask_yolo: MaskYolo) -> None:
    image_size = 256
    images = torch.rand(2, 3, image_size, image_size)
    masks = mask_yolo(images)


def test_mask_yolo_box_branch(mask_yolo: MaskYolo) -> None:
    image_size = 256
    images = torch.rand(2, 3, image_size, image_size)
    feats = mask_yolo.feats(images)
    box_feats = mask_yolo.box_feats(feats)
    res = mask_yolo.box_branch(box_feats)
    assert len(res) == len(mask_yolo.box_strides)
    for pred_box_batch, s in zip(res, mask_yolo.box_strides):
        assert pred_box_batch.shape == (2, 7, image_size // s, image_size // s)


def test_mask_yolo_local_mask_branch(mask_yolo: MaskYolo) -> None:
    out_channels = mask_yolo.backbone.out_channels
    strides = mask_yolo.strides
    image_size = 256
    image_batch = torch.rand(2, 3, image_size, image_size)
    box_batch = torch.tensor(
        [
            [10, 20, 30, 40],
            [10, 20, 30, 40],
        ]
    ).float()
    feats = mask_yolo.feats(image_batch)
    mask_feats = mask_yolo.mask_feats(feats)
    masks = mask_yolo.local_mask_branch(box_batch, mask_feats)


def test_criterion(
    mask_yolo: MaskYolo, targets: tuple[list[Tensor], list[Tensor]]
) -> None:
    criterion = Criterion(model=mask_yolo)
    images = torch.rand(2, 3, 128, 128)
    criterion(inputs=(images,), targets=targets)
