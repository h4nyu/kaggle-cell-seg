import torch
from cellseg.solo import Solo, Criterion, ToMasks
from cellseg.solo.adaptors import CentersToGridIndex, ToCategoryGrid, BatchAdaptor
from cellseg.backbones import EfficientNetFPN


def test_to_masks() -> None:
    grid_size = 4
    original_size = 8
    gt_masks = torch.zeros(2, original_size, original_size).bool()
    labels = torch.zeros(len(gt_masks))
    gt_masks[0, 0:3, 1:2] = True
    gt_masks[1, 3:4, 3:5] = True

    ba = BatchAdaptor(
        num_classes=1,
        grid_size=grid_size,
        original_size=original_size,
    )
    gt_mask_batch = [gt_masks]
    gt_label_batch = [labels]
    grids, gt_index_batch = ba(gt_mask_batch, gt_label_batch)
    to_masks = ToMasks()
    all_masks = torch.zeros(
        1, grid_size * grid_size, original_size, original_size
    ).bool()
    for gt_idx, all_idx in enumerate(gt_index_batch[0]):
        all_masks[0][all_idx] = gt_masks[gt_idx]

    mask_batch, label_batch = to_masks(grids, all_masks)
    assert (
        len(mask_batch) == len(gt_mask_batch) == len(label_batch) == len(gt_label_batch)
    )
    for masks, gt_masks in zip(mask_batch, gt_mask_batch):
        assert (masks ^ gt_masks).sum() == 0


def test_solo() -> None:
    image_batch = torch.rand(1, 3, 512, 512)
    backbone = EfficientNetFPN("efficientnet-b2")
    category_feat_range = (4, 6)
    mask_feat_range = (0, 4)
    num_classes = 2
    grid_size = 32

    solo = Solo(
        num_classes=num_classes,
        backbone=backbone,
        hidden_channels=64,
        grid_size=grid_size,
        category_feat_range=category_feat_range,
        mask_feat_range=mask_feat_range,
    )
    category_grid, all_masks = solo(image_batch)
    assert category_grid.shape == (1, num_classes, grid_size, grid_size)
    assert all_masks.shape == (1, grid_size ** 2, *image_batch.shape[2:])


def test_loss() -> None:
    batch_size = 1
    num_classes = 2
    grid_size = 10
    original_size = 100

    # net outputss
    pred_category_grids = (
        torch.zeros(batch_size, num_classes, grid_size, grid_size)
    )
    all_masks = (
        torch.zeros(batch_size, grid_size * grid_size, original_size, original_size)
    )

    # data adaptor outputs
    gt_category_grids = torch.zeros(batch_size, num_classes, grid_size, grid_size)
    gt_mask_batch = [torch.zeros(3, original_size, original_size)]
    mask_index_batch = [torch.tensor([1, 2, 3])]  # same len to mask_batch item

    loss = Criterion()
    loss_value, category_loss, mask_loss = loss(
        inputs=(pred_category_grids, all_masks),
        targets=(gt_category_grids, gt_mask_batch, mask_index_batch),
    )
    assert  category_loss + mask_loss == loss_value
    assert loss_value < 0.01
