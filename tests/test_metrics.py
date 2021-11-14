import pytest
import torch
from cellseg.metrics import seg_iou


@pytest.mark.parametrize(
    "inputs_len, expected",
    [
        (3, 0.5),
        (1, 0),
        (0, 0),
    ],
)
def test_seg_iou(inputs_len: int, expected: float) -> None:
    inputs = torch.zeros(1, 4, 4)
    inputs[0, 2, 0:inputs_len] = 1

    targets = torch.zeros(1, 4, 4)
    targets[0, 2, 3] = 1
    targets[0, 2, 2] = 1
    targets[0, 2, 1] = 1
    res = seg_iou(inputs, targets)
    assert res == expected
