import pytest
import torch
from cellseg.loss import SigmoidFocalLoss, DiceLoss


@pytest.mark.parametrize(
    "factor, expected",
    [
        (-100.0, 0.0),
        (100.0, 300.0),
    ],
)
def test_binary_focal_loss(factor: float, expected: float) -> None:
    loss = SigmoidFocalLoss()
    inputs = torch.ones(1, 1, 3, 3) * factor
    targets = torch.zeros(1, 1, 3, 3)
    res = loss(inputs, targets)
    assert round(res.item(), 1) == expected


@pytest.mark.parametrize(
    "factor, expected",
    [
        (-100.0, 1.0),
        (100.0, 0.0),
    ],
)
def test_dice_loss(factor: float, expected: float) -> None:
    loss = DiceLoss()
    size = 10
    inputs = torch.ones(1, size, size) * factor
    targets = torch.ones(1, size, size)
    res = loss(inputs, targets)
    assert round(res.item(), 1) == expected
