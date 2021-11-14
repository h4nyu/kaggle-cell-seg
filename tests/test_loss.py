import pytest
import torch
from cellseg.loss import SigmoidFocalLoss


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
