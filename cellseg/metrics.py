import torch
from torch import Tensor
import torch.nn.functional as F


class MaskIou:
    def __init__(self, use_batch: bool = False):
        self.use_batch = use_batch

    def _simple(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Tensor:
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

    def _batch(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Tensor:
        pred_masks = pred_masks.bool().view(pred_masks.shape[0], 1, -1)
        gt_masks = gt_masks.bool().view(gt_masks.shape[0], -1)
        intersection = (pred_masks & gt_masks).sum(dim=-1)
        union = (pred_masks | gt_masks).sum(dim=-1)
        iou_matrix = intersection / union
        iou_matrix = iou_matrix.nan_to_num(nan=0)
        return iou_matrix

    def __call__(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> Tensor:
        if self.use_batch:
            return self._batch(pred_masks, gt_masks)
        else:
            return self._simple(pred_masks, gt_masks)


class MaskAP:
    def __init__(
        self,
        reduce_size: int = 1,
        use_batch: bool = False,
        thresholds: list[float] = [
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ],
    ):
        self.thresholds = thresholds
        self.reduce_size = reduce_size
        self.mask_iou = MaskIou(use_batch=use_batch)
        self.num_samples = 0
        self.runing_value = 0.0

    @property
    def value(self) -> float:
        if self.num_samples == 0:
            return 0.0
        return self.runing_value / self.num_samples

    def precision_at(
        self, pred_masks: Tensor, gt_masks: Tensor, threshold: float
    ) -> float:
        iou_matrix = self.mask_iou(pred_masks, gt_masks)
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

    def accumulate(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
        value = self(pred_masks, gt_masks)
        self.num_samples += 1
        self.runing_value += value
        return value

    def accumulate_batch(
        self, pred_masks: list[torch.Tensor], gt_masks: list[torch.Tensor]
    ) -> None:
        for p, g in zip(pred_masks, gt_masks):
            self.accumulate(p, g)

    @torch.no_grad()
    def __call__(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> float:
        if self.reduce_size > 1:
            split_idx = pred_masks.shape[0]
            all_masks = F.interpolate(
                torch.cat([pred_masks, gt_masks]).unsqueeze(0).float(),
                scale_factor=1 / self.reduce_size,
            )[0].bool()
            pred_masks = all_masks[:split_idx]
            gt_masks = all_masks[split_idx:]
        running_p = 0.0
        for th in self.thresholds:
            running_p += self.precision_at(
                pred_masks=pred_masks, gt_masks=gt_masks, threshold=th
            )
        return running_p / len(self.thresholds)
