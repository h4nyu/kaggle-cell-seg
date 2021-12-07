import torch
import typing
from torch import Tensor
from torchvision.ops import box_convert
from torchvision.ops.boxes import box_iou


class ClosestAssign:
    """
    select k anchors whose center are closest to
    the center of ground-truth based on L2 distance.
    """

    def __init__(self, topk: int) -> None:
        self.topk = topk

    def __call__(self, anchor: Tensor, gt: Tensor) -> Tensor:
        device = anchor.device
        gt_count = gt.shape[0]
        anchor_count = anchor.shape[0]
        if gt_count == 0:
            return torch.zeros((0, gt_count), device=device)
        topk = min(anchor_count, self.topk)
        anchor_ctr = (
            ((anchor[:, :2] + anchor[:, 2:]) / 2.0)
            .view(anchor_count, 1, 2)
            .expand(
                anchor_count,
                gt_count,
                2,
            )
        )
        gt_ctr = gt[:, :2]
        matrix = ((anchor_ctr - gt_ctr) ** 2).sum(dim=-1).sqrt()
        _, matched_idx = torch.topk(matrix, topk, dim=0, largest=False)
        return matched_idx.t()


class ATSS:
    """
    Adaptive Training Sample Selection
    """

    def __init__(
        self,
        topk: int = 9,
    ) -> None:
        self.topk = topk
        self.assign = ClosestAssign(topk)

    def __call__(
        self,
        pred_boxes: Tensor,
        gt_boxes: Tensor,
    ) -> Tensor:  # [~topk, [gt_index, anchor_index]]
        device = pred_boxes.device
        matched_ids = self.assign(pred_boxes, gt_boxes)
        gt_count, _ = matched_ids.shape
        pred_count, _ = pred_boxes.shape
        pos_ids = torch.zeros(
            (
                gt_count,
                pred_count,
            ),
            device=device,
        )
        for i in range(gt_count):
            ids = matched_ids[i]
            matched_preds = pred_boxes[ids]
            ious = box_iou(matched_preds, gt_boxes[[i]]).view(-1)
            m_iou = ious.mean()
            s_iou = ious.std()
            th = m_iou + s_iou
            pos_ids[i, ids[ious > th]] = True
        return torch.nonzero(pos_ids, as_tuple=False)


class IoUAssign:
    def __init__(
        self,
        threshold: float = 0.7,
    ) -> None:
        self.threshold = threshold

    def __call__(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> Tensor:
        device = inputs.device
        if len(inputs) == 0 or len(targets) == 0:
            return torch.zeros(0, 2).to(device)
        iou_matrix = box_iou(inputs, targets)
        matched_value, matched_ids = torch.topk(iou_matrix, 1)
        pos_filter = (matched_value > self.threshold)[:, 0]
        input_index = torch.range(start=0, end=len(inputs) - 1).long().to(device)
        pair = torch.stack([input_index, matched_ids[:, 0]]).t()[pos_filter]
        return pair


class SimOTA:
    def __init__(
        self, topk: int, radius: float = 1.0, center_weight: float = 1.0
    ) -> None:
        self.topk = topk
        self.radius = radius
        self.center_weight = center_weight

    def candidates(
        self,
        anchor_points: Tensor,
        gt_boxes: Tensor,
        strides: Tensor,
    ) -> tuple[Tensor, Tensor]:
        gt_boxes = gt_boxes.unsqueeze(1)
        gt_centers = (gt_boxes[:, :, 0:2] + gt_boxes[:, :, 2:4]) / 2.0

        is_in_box = (  # grid cell inside gt-box
            (gt_boxes[:, :, 0] <= anchor_points[:, 0])
            & (anchor_points[:, 0] < gt_boxes[:, :, 2])
            & (gt_boxes[:, :, 1] <= anchor_points[:, 1])
            & (anchor_points[:, 1] < gt_boxes[:, :, 3])
        )  # [num_gts, num_proposals]
        gt_center_lbound = gt_centers - self.radius * strides.unsqueeze(1)
        gt_center_ubound = gt_centers + self.radius * strides.unsqueeze(1)

        is_in_center = (  # grid cell near gt-box center
            (gt_center_lbound[:, :, 0] <= anchor_points[:, 0])
            & (anchor_points[:, 0] < gt_center_ubound[:, :, 0])
            & (gt_center_lbound[:, :, 1] <= anchor_points[:, 1])
            & (anchor_points[:, 1] < gt_center_ubound[:, :, 1])
        )  # [num_gts, num_proposals]
        candidates = (is_in_box | is_in_center).any(dim=0)
        center_matrix = (is_in_box & is_in_center)[
            :, candidates
        ]  # [num_gts, num_fg_candidates]
        return candidates, center_matrix

    def __call__(
        self,
        anchor_points: Tensor,
        pred_boxes: Tensor,
        pred_scores: Tensor,
        gt_boxes: Tensor,
        strides: Tensor,
    ) -> Tensor:  # [gt_index, pred_index]
        device = pred_boxes.device
        gt_count = len(gt_boxes)
        pred_count = len(pred_boxes)
        if gt_count == 0 or pred_count == 0:
            return torch.zeros(0, 2).to(device)
        candidates, center_matrix = self.candidates(
            anchor_points=anchor_points,
            gt_boxes=gt_boxes,
            strides=strides,
        )
        score_matrix = pred_scores[candidates].expand(gt_count, -1)
        iou_matrix = box_iou(gt_boxes, pred_boxes[candidates])
        matrix = score_matrix + iou_matrix + center_matrix * self.center_weight
        topk = min(self.topk, pred_count)
        topk_ious, _ = torch.topk(iou_matrix, topk, dim=1)
        dynamic_ks = topk_ious.sum(1).int().clamp(min=1)
        matching_matrix = torch.zeros((gt_count, pred_count), dtype=torch.long)
        candidate_idx = candidates.nonzero().view(-1)
        for (row, dynamic_topk, matching_row) in zip(
            matrix, dynamic_ks, matching_matrix
        ):
            _, pos_idx = torch.topk(row, k=dynamic_topk)
            matching_row[candidate_idx[pos_idx]] = 1
        return matching_matrix.nonzero()
