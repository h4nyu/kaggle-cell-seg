import pytest
import torch
from cellseg.loss import SigmoidFocalLoss, MaskLoss, DiceLoss


@pytest.mark.parametrize(
    "factor, expected",
    [
        (-100.0, 0.0),
        (100.0, 300.0),
    ],
)
def test_binary_focal_loss(factor: float, expected: float) -> None:
    loss = SigmoidFocalLoss()
    source = torch.ones(1, 1, 3, 3) * factor
    target = torch.zeros(1, 1, 3, 3)
    res = loss(source, target)
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


def test_mask_loss() -> None:
    loss = MaskLoss()

    pred_masks = torch.zeros(100, 3, 3)
    gt_masks = torch.ones(2, 3, 3)
    mask_index = torch.tensor([1, 2])
    res = loss(
        pred_masks=pred_masks,
        mask_index=mask_index,
        gt_masks=gt_masks,
    )
    # assert round(res.item(), 1) == expected
