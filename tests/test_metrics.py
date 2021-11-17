import pytest
import torch
from torch import Tensor
from cellseg.metrics import mask_iou, precision_at, precision


@pytest.mark.parametrize(
    "inputs_len, expected",
    [
        (3, [0.5, 0.0]),
        (1, [0, 0]),
        (0, [0, 0]),
    ],
)
def test_mask_iou(inputs_len: int, expected: list[float]) -> None:
    pred_masks = torch.zeros(3, 4, 4)
    pred_masks[0, 2, 0:inputs_len] = 1

    gt_masks = torch.zeros(2, 4, 4)
    gt_masks[0, 2, 3] = 1
    gt_masks[0, 2, 2] = 1
    gt_masks[0, 2, 1] = 1
    res = mask_iou(pred_masks, gt_masks)
    assert res.shape == (len(pred_masks), len(gt_masks))
    for i, v in enumerate(expected):
        assert res[0][i] == v


def test_large_mask_iou() -> None:
    pred_masks = torch.zeros(10, 500, 500)
    gt_masks = torch.zeros(400, 500, 500)
    res = mask_iou(pred_masks, gt_masks)
    assert res.shape == (len(pred_masks), len(gt_masks))


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
