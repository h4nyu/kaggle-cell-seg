import pytest
import torch
from torch import Tensor
from cellseg.metrics import seg_iou, precision_at, precision


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


def test_seg_iou_masks() -> None:
    inputs = torch.zeros(1, 4, 4)
    inputs[0, 2, 0:2] = 1

    targets = torch.zeros(3, 4, 4)
    targets[0, 2, 3] = 1
    targets[0, 2, 2] = 1
    targets[0, 2, 1] = 1
    res = seg_iou(inputs, targets)


def test_precision_at() -> None:
    inputs = torch.zeros(2, 4, 4)
    inputs[0, 0:2, 0:2] = 1
    inputs[1, 2, 1] = 1

    targets = torch.zeros(3, 4, 4)
    targets[0, 0, 0] = 1
    targets[1, 1, 1] = 1
    targets[2, 2, 2] = 1
    res = precision_at(inputs, targets, 0.2)
    assert res == 1 / 4


def test_precision() -> None:
    masks = torch.load("/app/data/masks-0030fd0e6378.pth")
    res = precision(
        pred_masks=masks,
        gt_masks=masks,
    )
    assert res == 1.0
