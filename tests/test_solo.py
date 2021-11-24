import torch
import torch.nn as nn
from torch import Tensor
from cellseg.backbones import EfficientNetFPN
from torch.utils.data import Subset, DataLoader
import os
import torch.nn.functional as F
from cellseg.data import (
    CellTrainDataset,
    collate_fn,
)
from cellseg.solo import (
    ToCategoryGrid,
    MasksToCenters,
    CentersToGridIndex,
    BatchAdaptor,
    Solo,
    Criterion,
    ToMasks,
    MatrixNms,
    PatchInferenceStep,
)
from cellseg.utils import draw_save, ToPatches


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
                4,
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
    category_grid, matched = to_grid(
        centers=centers,
        labels=labels,
    )
    assert category_grid.shape == (num_classes, grid_size, grid_size)
    assert category_grid.sum() == len(matched)
    assert category_grid[0, 2, 1] == 1
    assert category_grid[0, 2, 4] == 1
    assert matched.tolist() == [[17, 0], [2 * grid_size + 4, 1]]


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
    category_grids, index_batch = a(
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
    grids, gt_index_batch = ba(gt_mask_batch, gt_label_batch)
    grids[0, 0, 0, 0] = 0.9
    to_masks = ToMasks()
    all_masks = torch.zeros(
        1, grid_size * grid_size, original_size, original_size
    ).bool()
    for gt_idx, all_idx in enumerate(gt_index_batch[0]):
        all_masks[0][all_idx] = gt_masks[gt_idx]

    mask_batch, label_batch, score_batch = to_masks(grids, all_masks)
    assert (
        len(mask_batch)
        == len(gt_mask_batch)
        == len(label_batch)
        == len(gt_label_batch)
        == len(score_batch)
    )
    assert score_batch[0][0] == 0.9
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
    mask_index_batch = [torch.tensor([[0, 0], [1, 1], [2, 2]])]

    loss = Criterion()
    loss_value, category_loss, mask_loss = loss(
        inputs=(pred_category_grids, all_masks),
        targets=(
            gt_category_grids,
            gt_mask_batch,
            mask_index_batch,
        ),
    )
    assert category_loss + mask_loss == loss_value
    assert loss_value < 0.01


def test_nms() -> None:
    nms = MatrixNms()
    image_size = 8
    masks = torch.zeros(2, image_size, image_size)
    masks[0, 2:4, 2:4] = 1.0
    masks[1, 2:5, 2:5] = 1.0

    cate_labels = torch.zeros(len(masks), dtype=torch.long)
    cate_scores = torch.tensor([0.9, 0.4], dtype=torch.float)
    res = nms(masks, cate_labels, cate_scores)
    assert res.shape == cate_scores.shape


def test_inference_step() -> None:
    backbone = EfficientNetFPN("efficientnet-b0")
    category_feat_range = (4, 6)
    mask_feat_range = (0, 4)
    num_classes = 1
    grid_size = 16
    patch_size = 128

    solo = Solo(
        num_classes=num_classes,
        backbone=backbone,
        hidden_channels=64,
        grid_size=grid_size,
        category_feat_range=category_feat_range,
        mask_feat_range=mask_feat_range,
    )
    to_masks = ToMasks()
    batch_adaptor = BatchAdaptor(
        num_classes=num_classes,
        grid_size=grid_size,
        original_size=patch_size,
    )
    to_patches = ToPatches(patch_size=patch_size)
    inference_step = PatchInferenceStep(
        model=solo,
        batch_adaptor=batch_adaptor,
        to_masks=to_masks,
        patch_size=patch_size,
    )
    dataset = CellTrainDataset(
        img_dir="data",
        train_csv="data/annotation.csv",
    )
    sample = dataset[0]
    assert sample is not None
    image = sample["image"]
    masks = sample["masks"]
    labels = sample["labels"]
    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        # **cfg.validation_loader,
    )
    for batch in loader:
        inference_step(batch[0])
    # ...
