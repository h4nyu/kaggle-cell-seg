import torch
from torch import Tensor
import numpy as np


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Tensor:
    iou_rows = []
    pred_masks = pred_masks.bool().view(pred_masks.shape[0], -1)
    gt_masks = gt_masks.bool().view(gt_masks.shape[0], -1)
    for pred_mask in pred_masks:
        intersection = (gt_masks & pred_mask).sum(dim=-1)
        union = (gt_masks | pred_mask).sum(dim=-1)
        iou_row = intersection / union
        iou_rows.append(iou_row)
    iou_matrix = torch.stack(iou_rows).nan_to_num(nan=0)
    return iou_matrix


def precision_at(pred_masks: Tensor, gt_masks: Tensor, threshold: float) -> float:
    iou_matrix = mask_iou(pred_masks, gt_masks)
    num_preds, num_gt = iou_matrix.shape
    fp = torch.ones(num_gt, dtype=torch.bool)
    for ious in iou_matrix:
        iou, gt_idx = ious.max(dim=0)
        if iou >= threshold:
            fp[gt_idx] = False
    fp_count = fp.sum()
    tp_count = num_gt - fp_count
    res = tp_count / (num_gt + num_preds - tp_count)
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
