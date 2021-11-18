import pytest
import torch
from cellseg.loss import FocalLoss, DiceLoss


@pytest.mark.parametrize(
    "factor, expected",
    [
        (0.01, 0.0),
        (0.99, 3.8),
    ],
)
def test_binary_focal_loss(factor: float, expected: float) -> None:
    loss = FocalLoss()
    inputs = torch.ones(1, 1, 3, 3) * factor
    targets = torch.zeros(1, 1, 3, 3)
    res = loss(inputs, targets)
    assert round(res.item(), 1) == expected


@pytest.mark.parametrize(
    "factor, expected",
    [
        (0.1, 0.8),
        (0.9, 0.1),
    ],
)
def test_dice_loss(factor: float, expected: float) -> None:
    loss = DiceLoss()
    size = 10
    inputs = torch.ones(1, size, size) * factor
    targets = torch.ones(1, size, size)
    res = loss(inputs, targets)
    assert round(res.item(), 1) == expected
