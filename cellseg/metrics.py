import os
import torch
from torch import Tensor
import numpy as np


def seg_iou(pred: torch.Tensor, targets: torch.Tensor) -> Tensor:
    pred = pred.bool()
    targets = targets.bool()
    intersection = (pred & targets).sum(dim=[1, 2])
    union = (pred | targets).sum(dim=[1, 2])
    iou = intersection / union
    return iou


def precision_at(pred_masks: Tensor, gt_masks: Tensor, threshold: float) -> float:
    fp = torch.ones(len(gt_masks), dtype=torch.bool)
    for mask_id, mask in enumerate(pred_masks):
        iou = seg_iou(mask, gt_masks) > threshold
        if iou.sum() != 0:
            gt_mask_idx = iou.short().argmax()
            fp[gt_mask_idx] = False
    fp_count = fp.sum()
    tp_count = len(gt_masks) - fp_count
    fn_count = len(pred_masks) - tp_count
    res = tp_count / (len(gt_masks) + len(pred_masks) - tp_count)
    return res.item()


def precision(
    pred_masks: Tensor,
    gt_masks: Tensor,
    thresholds: list[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
) -> float:
    precisions = [
        precision_at(pred_masks=pred_masks, gt_masks=gt_masks, threshold=threshold)
        for threshold in thresholds
    ]
    print(precisions)
    return np.mean(precisions)
