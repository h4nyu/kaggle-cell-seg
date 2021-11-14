import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class SigmoidFocalLoss:
    def __init__(
        self,
        alpha: float = 3.0,
        gamma: float = 2.0,
        reduction: str = "mean",
        smooth: float = 1e-6,  # set '1e-4' when train with FP16
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

    def __call__(self, source: Tensor, target: Tensor) -> Tensor:
        prob = torch.sigmoid(source)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = (
            -self.alpha * neg_weight * F.logsigmoid(-source)
        )  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        if self.reduction == "mean":
            loss = loss.mean()
        return loss


class DiceLoss:
    def __init__(self, smooth: float = 1.0) -> None:
        self.smooth = smooth

    def __call__(self, inputs: Tensor, targets: Tensor):

        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class MaskLoss:
    def __init__(self) -> None:
        ...

    def __call__(
        self, pred_masks: Tensor, mask_index: Tensor, gt_masks: Tensor
    ) -> None:
        matched_masks = pred_masks[mask_index]
        print(matched_masks.shape)
        ...
