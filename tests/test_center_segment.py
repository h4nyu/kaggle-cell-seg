import pytest
from cellseg.center_segment import (
    CenterSegment,
    TrainStep,
    Criterion,
    BatchAdaptor,
)
from cellseg.backbones import EfficientNetFPN
from cellseg.util import ToDevice
import torch
import torch.optim as optim


def test_center_segment() -> None:
    image_batch = torch.rand(2, 3, 512, 512)
    backbone = EfficientNetFPN("efficientnet-b0")
    num_classes = 1
    box_size = 32
    category_feat_range = (3, 6)

    model = CenterSegment(
        num_classes=num_classes,
        backbone=backbone,
        category_feat_range=category_feat_range,
        hidden_channels=64,
        box_size=box_size,
    )
    model(image_batch)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpu")
def test_train_step() -> None:
    backbone = EfficientNetFPN("efficientnet-b0")
    num_classes = 1
    original_size = 384
    category_feat_range = (0, 3)
    box_size = 32
    device = "cuda"

    image_batch = torch.rand(1, 3, original_size, original_size)
    gt_masks = torch.zeros(2, original_size, original_size).bool()
    gt_masks[0, 0:50, 0:60] = True
    gt_masks[1, 110:120, 130:150] = True
    gt_mask_batch = [gt_masks]
    gt_label_batch = [torch.zeros(len(gt_masks))]
    to_device = ToDevice(device)
    batch = to_device(image_batch, gt_mask_batch, gt_label_batch)

    model = CenterSegment(
        num_classes=num_classes,
        backbone=backbone,
        category_feat_range=category_feat_range,
        hidden_channels=64,
        box_size=box_size,
    ).to(device)
    optimizer = optim.Adam(model.parameters())
    batch_adaptor = BatchAdaptor(
        num_classes=num_classes,
        original_size=original_size,
        grid_size=original_size,
        box_size=box_size,
    )

    criterion = Criterion(
        box_size=box_size,
    )
    train_step = TrainStep(
        model=model,
        criterion=criterion,
        batch_adaptor=batch_adaptor,
        optimizer=optimizer,
    )
    train_step(batch)
