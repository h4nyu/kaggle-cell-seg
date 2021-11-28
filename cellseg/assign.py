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
        _, matched_idx = torch.topk(matrix, self.topk, dim=0, largest=False)
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
        anchors: Tensor,
        gt: Tensor,
    ) -> Tensor:
        device = anchors.device
        matched_ids = self.assign(anchors, gt)
        gt_count, _ = matched_ids.shape
        anchor_count, _ = anchors.shape
        pos_ids = torch.zeros(
            (
                gt_count,
                anchor_count,
            ),
            device=device,
        )
        for i in range(gt_count):
            ids = matched_ids[i]
            matched_anchors = anchors[ids]
            ious = box_iou(matched_anchors, gt[[i]]).view(-1)
            m_iou = ious.mean()
            s_iou = ious.std()
            th = m_iou + s_iou
            pos_ids[i, ids[ious > th]] = True
        return torch.nonzero(pos_ids, as_tuple=False)


class IoUAssign:
    """
    Adaptive Training Sample Selection
    """

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
