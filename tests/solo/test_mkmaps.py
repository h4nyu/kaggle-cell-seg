import torch.nn as nn
import os
import torch
from torchvision.ops import masks_to_boxes
import torch.nn.functional as F
from cellseg.config import ROOT_PATH
from cellseg.solo.mkmaps import MkGaussianMaps
from cellseg.data import draw_save


@torch.no_grad()
def test_mkmaps() -> None:
    masks = torch.load("data/masks-0030fd0e6378.pth")
    boxes = masks_to_boxes(masks)
    labels = torch.zeros(boxes.shape[0])
    mkmaps = MkGaussianMaps(num_classes=1)
    heatmap = mkmaps(
        box_batch=[boxes],
        label_batch=[labels],
        hw=(64, 64),
        original_hw=masks.shape[1:],
    )
    draw_save(heatmap[0], os.path.join(ROOT_PATH, "test-heatmap.png"))
    assert heatmap.sum() == len(boxes)
