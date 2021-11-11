import torch.nn as nn
import os
import torch
import torch.nn.functional as F
from cellseg.config import ROOT_PATH
from cellseg.solo.mkmaps import MkGaussianMaps
from cellseg.data import draw_save


@torch.no_grad()
def test_mkmaps() -> None:
    masks = torch.load("data/masks-0030fd0e6378.pth")
    labels = torch.zeros(masks.shape[0])
    mkmaps = MkGaussianMaps(num_classes=1)
    heatmap = mkmaps(
        mask_batch=[masks],
        label_batch=[labels],
        hw=(64, 64),
        original_hw=masks.shape[1:],
    )
    draw_save(heatmap[0], os.path.join(ROOT_PATH, "test-heatmap.png"))
    assert heatmap.sum() == len(masks)


@torch.no_grad()
def test_mkmaskmaps() -> None:
    masks = torch.load("data/masks-0030fd0e6378.pth")[:3]
    labels = torch.zeros(masks.shape[0])
    mkmaps = MkGaussianMaps(num_classes=1)
    res = mkmaps._mkmaskmap(grid_size=(64, 64), out_size=masks.shape[1:], masks=masks)
    assert res.sum() == masks.sum()
