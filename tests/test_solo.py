import torch
import torch.nn as nn
from torch import Tensor
from cellseg.backbones import EfficientNetFPN
import os
import torch.nn.functional as F
from cellseg.solo import (
    ToCategoryGrid,
    MasksToCenters,
    CentersToGridIndex,
    BatchAdaptor,
    Solo,
    Criterion,
    ToMasks,
)
from cellseg.util import draw_save


def test_center_to_grid_index() -> None:
    centers = Tensor(
        [
            [
                2,
                1,
            ],
        ]
    )
    index = CentersToGridIndex(grid_size=64)(centers)
    assert index.dtype == torch.long
    assert index[0] == 1 * 64 + 2


def test_category_adaptor() -> None:
    centers = Tensor(
        [
            [
                1,
                2,
            ],
            [
                1,
                2,
            ],
            [
                1,
                2,
            ],
        ]
    )
    labels = torch.zeros(len(centers))
    grid_size = 8
    num_classes = 1
    to_grid = ToCategoryGrid(
        num_classes=num_classes,
        grid_size=grid_size,
    )
    category_grid, mask_index, labels = to_grid(
        centers=centers,
        labels=labels,
    )
    assert category_grid.shape == (num_classes, grid_size, grid_size)
    assert category_grid.sum() == len(mask_index) == len(labels)
    assert category_grid[0, 2, 1] == 1
    assert mask_index.tolist() == [17]


def test_batch_adaptor() -> None:
    original_size = 100
    grid_size = 10
    batch_size = 2
    num_classes = 2
    a = BatchAdaptor(num_classes, grid_size, original_size)
    mask_batch = [
        torch.ones(2, original_size, original_size),
        torch.ones(3, original_size, original_size),
    ]
    label_batch = [torch.zeros(len(i)) for i in mask_batch]
    category_grids, index_batch, label_batch = a(
        mask_batch=mask_batch,
        label_batch=label_batch,
    )
    assert category_grids.shape == (batch_size, num_classes, grid_size, grid_size)
    assert len(index_batch) == batch_size
    for index, masks in zip(index_batch, mask_batch):
        assert index.shape[0] == masks.shape[0]
        print(index.shape, masks.shape)


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
    grids, gt_index_batch, gt_label_batch = ba(gt_mask_batch, gt_label_batch)
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
    pred_category_grids = torch.zeros(batch_size, num_classes, grid_size, grid_size)
    all_masks = torch.zeros(
        batch_size, grid_size * grid_size, original_size, original_size
    )

    # data adaptor outputs
    gt_category_grids = torch.zeros(batch_size, num_classes, grid_size, grid_size)
    gt_mask_batch = [torch.zeros(3, original_size, original_size)]
    mask_index_batch = [torch.tensor([1, 2, 3])]  # same len to mask_batch item
    filter_index_batch = [torch.tensor([1, 2, 3])]  # same len to mask_batch item

    loss = Criterion()
    loss_value, category_loss, mask_loss = loss(
        inputs=(pred_category_grids, all_masks),
        targets=(
            gt_category_grids,
            gt_mask_batch,
            mask_index_batch,
            filter_index_batch,
        ),
    )
    assert category_loss + mask_loss == loss_value
    assert loss_value < 0.01
