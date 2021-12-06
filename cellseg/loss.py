import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_area, box_iou
from typing import Optional


class FocalLoss(nn.Module):
    """
    Modified focal loss
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 2.0,
        eps: float = 5e-4,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """
        pred: 0-1 [B, C,..]
        gt: 0-1 [B, C,..]
        """
        alpha = self.alpha
        beta = self.beta
        eps = self.eps
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        pos_mask = gt.eq(1).float()
        neg_mask = gt.lt(1).float()
        pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos_mask
        pos_loss = pos_loss.sum() / pos_mask.sum().clamp(min=1.0)

        neg_weight = (1 - gt.float()) ** beta
        neg_loss = neg_weight * (-(pred ** alpha) * torch.log(1 - pred) * neg_mask)
        neg_loss = neg_loss.sum() / neg_mask.sum().clamp(min=1.0)
        loss = pos_loss + neg_loss
        return loss


class DiceLoss:
    def __init__(self, smooth: float = 1.0) -> None:
        self.smooth = smooth

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class IoU:
    def __call__(self, boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union


class GIoU:
    def __init__(self) -> None:
        self.iou = IoU()

    def __call__(self, src: Tensor, tgt: Tensor) -> Tensor:
        iou, union = self.iou(src, tgt)
        lt = torch.min(src[:, None, :2], tgt[:, :2])
        rb = torch.max(src[:, None, 2:], tgt[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]
        return 1 - iou + (area - union) / area


class DIoU:
    def __init__(self) -> None:
        self.iou = IoU()
        self.eps = 1e-4

    def __call__(self, src: Tensor, tgt: Tensor) -> Tensor:
        iou, _ = self.iou(src, tgt)
        s_ctr = (src[:, None, :2] + src[:, None, 2:]) / 2
        t_ctr = (tgt[:, :2] + tgt[:, 2:]) / 2

        lt = torch.min(src[:, None, :2], tgt[:, :2])
        rb = torch.max(src[:, None, 2:], tgt[:, 2:])

        diagnol = torch.pow((rb - lt).clamp(min=self.eps), 2).sum(dim=-1)
        ctr_dist = torch.pow(s_ctr - t_ctr, 2).sum(dim=-1)

        return 1 - iou + ctr_dist / diagnol


class IoULoss:
    def __init__(self, size_average: bool = True) -> None:
        self.size_average = size_average

    def __call__(self, boxes1: Tensor, boxes2: Tensor) -> tuple[Tensor, Tensor]:
        device = boxes1.device
        if len(boxes1) == 0 and len(boxes2) == 0:
            return torch.tensor(0.0, device=device), torch.zeros(0, device=device)
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, 0] * wh[:, 1]  # [N,M]

        union = area1 + area2 - inter

        iou = inter / union

        if self.size_average:
            iou = iou.mean()
        return 1 - iou, union


class DIoULoss:
    def __init__(self, size_average: bool = True) -> None:
        self.iouloss = IoULoss(size_average=size_average)
        self.size_average = size_average

    def __call__(self, src: Tensor, tgt: Tensor) -> Tensor:
        device = src.device
        if len(src) == 0 and len(tgt) == 0:
            if self.size_average:
                return torch.tensor(0.0, device=device)
            return torch.zeros(0, device=device)
        iouloss, _ = self.iouloss(src, tgt)
        s_ctr = (src[:, :2] + src[:, 2:]) / 2
        t_ctr = (tgt[:, :2] + tgt[:, 2:]) / 2
        lt = torch.min(src[:, :2], tgt[:, :2])
        rb = torch.max(src[:, 2:], tgt[:, 2:])

        ctr_loss = torch.pow(s_ctr - t_ctr, 2).sum(dim=-1) / torch.pow(
            (rb - lt).clamp(min=0), 2
        ).sum(dim=-1)
        if self.size_average:
            ctr_loss = ctr_loss.mean()

        return iouloss + ctr_loss


class SCALoss:
    def __init__(
        self,
        alpha: float = 1.0,
    ) -> None:
        self.alpha = alpha

    def __call__(self, inputs: Tensor, targets: Tensor) -> None:
        area1 = box_area(inputs)
        area2 = box_area(targets)

        lt = torch.max(inputs[:, None, :2], targets[:, :2])  # [N,M,2]
        rb = torch.min(inputs[:, None, 2:], targets[:, 2:])  # [N,M,2]
        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
        print(inter)



class CIoULoss:
    def __init__(self, eps:float=1e-15) -> None:
        self.eps = eps


    def box_iou(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)
        lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])
        rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1 + area2 - inter
        return inter / (union + self.eps)

    def _compute_aspect_factor(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        w1 = boxes1[..., 2] - boxes1[..., 0]
        h1 = boxes1[..., 3] - boxes1[..., 1]
        theta1 = torch.atan(w1 / (h1 + self.eps))
        w2 = boxes2[..., 2] - boxes2[..., 0]
        h2 = boxes2[..., 3] - boxes2[..., 1]
        theta2 = torch.atan(w2 / (h2 + self.eps))
        v = (4 / (math.pi ** 2)) * ((theta2 - theta1) ** 2)
        return v

    def _compute_distance_factor(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        center1 = (boxes1[..., 2:] + boxes1[..., :2]) * 0.5
        center2 = (boxes2[..., 2:] + boxes2[..., :2]) * 0.5
        center_distance = ((center2 - center1) ** 2).sum(dim=-1)
        convex_lt = torch.minimum(boxes1[..., :2], boxes2[..., :2])
        convex_rb = torch.maximum(boxes1[..., 2:], boxes2[..., 2:])
        convex_diag = ((convex_rb - convex_lt) ** 2).sum(dim=-1)
        res = center_distance / (convex_diag + self.eps)
        return res

    def box_ciou(
        self, boxes1: Tensor, boxes2: Tensor
    ) -> Tensor:
        boxes1, boxes2 = boxes1.float(), boxes2.float()  # force fp32
        iou = self.box_iou(boxes1, boxes2)
        u = self._compute_distance_factor(boxes1, boxes2)
        v = self._compute_aspect_factor(boxes1, boxes2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)
        return iou - (u + alpha * v)

    def __call__(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        ciou = self.box_ciou(boxes1, boxes2)
        return 1.0 - ciou.mean()
