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

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        prob = torch.sigmoid(inputs)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = (
            -self.alpha * neg_weight * F.logsigmoid(-inputs)
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
