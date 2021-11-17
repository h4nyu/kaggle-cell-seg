import pytest
import torch
from torch import Tensor
from cellseg.metrics import MaskIou, MaskAP


@pytest.mark.parametrize(
    "inputs_len, expected",
    [
        (3, [0.5, 0.0]),
        (1, [0, 0]),
        (0, [0, 0]),
    ],
)
def test_mask_iou(inputs_len: int, expected: list[float]) -> None:
    mask_iou = MaskIou()
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


def test_simple_batch_is_same() -> None:
    pred_masks = torch.zeros(3, 4, 4)
    pred_masks[0, 2, 0:2] = 1

    gt_masks = torch.zeros(2, 4, 4)
    gt_masks[0, 2, 3] = 1
    gt_masks[0, 2, 2] = 1
    gt_masks[0, 2, 1] = 1
    simple = MaskIou(use_batch=False)
    batch = MaskIou(use_batch=True)
    for s, b in zip(simple(pred_masks, gt_masks), batch(pred_masks, gt_masks)):
        for s0, b0 in zip(s, b):
            assert s0 == b0


def test_large_mask_iou() -> None:
    mask_iou = MaskIou(use_batch=False)
    pred_masks = torch.zeros(10, 500, 500)
    gt_masks = torch.zeros(400, 500, 500)
    res = mask_iou(pred_masks, gt_masks)
    assert res.shape == (len(pred_masks), len(gt_masks))



def test_precision_at() -> None:
    mask_ap = MaskAP()
    inputs = torch.zeros(2, 4, 4)
    inputs[0, 0:2, 0:2] = 1
    inputs[1, 2, 1] = 1

    targets = torch.zeros(3, 4, 4)
    targets[0, 0, 0] = 1
    targets[1, 1, 1] = 1
    targets[2, 2, 2] = 1
    res = mask_ap.precision_at(inputs, targets, 0.2)
    assert res == 1 / 4


def test_precision() -> None:
    masks = torch.load("/app/data/masks-0030fd0e6378.pth")
    mask_ap = MaskAP()
    mask_ap.accumulate(
        pred_masks=masks,
        gt_masks=masks,
    )
    assert mask_ap.value == 1.0

def test_reduce_size() -> None:
    masks = torch.load("/app/data/masks-0030fd0e6378.pth")
    reduce_mask_ap = MaskAP(reduce_size=8)
    full_mask_ap = MaskAP()
    pred_masks = masks.clone()
    pred_masks[0] = False
    full_mask_ap.accumulate(
        pred_masks=pred_masks,
        gt_masks=masks,
    )
    reduce_mask_ap.accumulate(
        pred_masks=pred_masks,
        gt_masks=masks,
    )
    assert reduce_mask_ap.value == full_mask_ap.value

