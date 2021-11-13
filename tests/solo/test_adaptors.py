import torch.nn as nn
from torch import Tensor
import os
import torch
import torch.nn.functional as F
from cellseg.config import ROOT_PATH
from cellseg.solo.adaptors import (
    ToCategoryGrid,
    MasksToCenters,
    CentersToGridIndex,
)
from cellseg.data import draw_save


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
                2,
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
    category_grid, mask_index = to_grid(
        centers=centers,
        labels=labels,
    )
    assert category_grid.shape == (num_classes, grid_size, grid_size)
    assert category_grid.sum() == len(centers)
    assert category_grid[0, 2, 2] == 1
    assert category_grid[0, 2, 1] == 1
    assert mask_index.tolist() == [17, 18]
