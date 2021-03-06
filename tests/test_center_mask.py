import torch
from cellseg.center_mask import CenterMask, BatchAdaptor, ToMasks, TrainStep
from cellseg.necks import CSPNeck
from cellseg.backbones import EfficientNetFPN


def test_model() -> None:
    num_classes = 2
    mask_size = 16
    images = torch.rand(1, 3, 512, 512)
    backbone = EfficientNetFPN("efficientnet-b2")
    category_feat_range = (4, 6)
    num_classes = 2
    neck = CSPNeck(
        in_channels=backbone.out_channels,
        out_channels=backbone.out_channels,
        strides=backbone.strides,
    )

    model = CenterMask(
        hidden_channels=64,
        backbone=backbone,
        neck=neck,
        mask_size=mask_size,
        num_classes=num_classes,
        category_feat_range=category_feat_range,
    )
    category_grids, size_grids, offset_grids, mask_grids, sliency_masks = model(images)
    assert category_grids.shape == (1, num_classes, 32, 32)
    assert size_grids.shape == (1, 2, 32, 32)
    assert offset_grids.shape == (1, 2, 32, 32)
    assert mask_grids.shape == (1, mask_size ** 2, 32, 32)
    assert sliency_masks.shape == (1, 1, 512, 512)


def test_batch() -> None:
    patch_size = 16
    num_classes = 2
    grid_size = 8
    mask_size = 4
    masks = torch.zeros(2, patch_size, patch_size).bool()
    labels = torch.zeros(len(masks)).long()
    masks[0, 0:6, 1:3] = True
    masks[1, 4:9, 3:6] = True
    mask_batch = [masks]
    label_batch = [labels]
    batch_adaptor = BatchAdaptor(
        num_classes=num_classes,
        grid_size=grid_size,
        patch_size=patch_size,
        mask_size=mask_size,
    )
    (
        category_grids,
        size_grids,
        offset_grids,
        mask_grids,
        sliency_masks,
        pos_masks,
    ) = batch_adaptor(mask_batch=mask_batch, label_batch=label_batch)
    assert pos_masks.shape == (1, 1, grid_size, grid_size)
    assert pos_masks.dtype == torch.bool
    assert pos_masks.sum() == len(masks)
    assert pos_masks[0, 0, 1, 0] == True
    assert pos_masks[0, 0, 3, 2] == True
    assert category_grids.shape == (1, num_classes, grid_size, grid_size)
    assert category_grids[0, 0, 1, 0] == 1
    assert category_grids[0, 0, 3, 2] == 1

    assert size_grids.shape == (1, 2, grid_size, grid_size)
    assert size_grids[0, :, 1, 0].tolist() == [0.0625, 0.3125]
    assert size_grids[0, :, 3, 2].tolist() == [0.125, 0.25]

    assert offset_grids.shape == (1, 2, grid_size, grid_size)
    assert offset_grids[0, :, 1, 0].tolist() == [0.75, 0.25]
    assert offset_grids[0, :, 3, 2].tolist() == [0.0, 0.0]

    assert mask_grids.shape == (1, mask_size ** 2, grid_size, grid_size)
    assert mask_grids[0, :, 1, 0].sum() != 0.0
    assert mask_grids[0, :, 3, 2].sum() != 0.0

    assert sliency_masks.shape == (1, 1, patch_size, patch_size)


def test_to_masks() -> None:
    patch_size = 16
    num_classes = 2
    grid_size = 8
    mask_size = 4
    masks = torch.zeros(2, patch_size, patch_size).bool()
    labels = torch.zeros(len(masks)).long()
    masks[0, 0:6, 1:3] = True
    masks[1, 4:9, 3:6] = True
    gt_mask_batch = [masks]
    gt_label_batch = [labels]
    batch_adaptor = BatchAdaptor(
        num_classes=num_classes,
        grid_size=grid_size,
        patch_size=patch_size,
        mask_size=mask_size,
    )
    (
        category_grids,
        size_grids,
        offset_grids,
        mask_grids,
        sliency_masks,
        pos_masks,
    ) = batch_adaptor(mask_batch=gt_mask_batch, label_batch=gt_label_batch)
    to_masks = ToMasks()
    mask_batch, label_batch, score_batch = to_masks(
        category_grids,
        size_grids,
        offset_grids,
        mask_grids,
        sliency_masks,
    )
    assert len(mask_batch) == len(label_batch) == len(score_batch) == 1
    assert (mask_batch[0] ^ gt_mask_batch[0]).sum() == 0
