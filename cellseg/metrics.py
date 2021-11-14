import os
import torch
from torch import Tensor
import numpy as np


def seg_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Tensor:
    pred_masks = pred_masks.bool().view(pred_masks.shape[0], 1, *pred_masks.shape[1:])
    gt_masks = gt_masks.bool()
    intersection = (pred_masks & gt_masks).sum(dim=[2, 3])
    union = (pred_masks | gt_masks).sum(dim=[2, 3])
    iou_matrix = intersection / union
    iou_matrix = iou_matrix.nan_to_num(nan=0)
    return iou_matrix


def precision_at(pred_masks: Tensor, gt_masks: Tensor, threshold: float) -> float:
    fp = torch.ones(len(gt_masks), dtype=torch.bool)
    iou_matrix = seg_iou(pred_masks, gt_masks)
    for iou_per_pred_mask in iou_matrix:
        if (iou_per_pred_mask > threshold).sum() != 0:
            gt_mask_idx = iou_per_pred_mask.short().argmax()
            fp[gt_mask_idx] = False
    fp_count = fp.sum()
    tp_count = len(gt_masks) - fp_count
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
    return np.mean(precisions)
