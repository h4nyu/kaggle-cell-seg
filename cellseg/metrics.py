import os
import torch
from torch import Tensor


def seg_iou(inputs: torch.Tensor, targets: torch.Tensor) -> Tensor:
    inputs = inputs.bool()
    targets = targets.bool()

    intersection = (inputs & targets).sum()
    if intersection == 0:
        return intersection
    union = (inputs | targets).sum()
    iou = intersection / union
    return iou
