import pytest
import torch
from cellseg.assign import IoUAssign, ATSS, SimOTA


def test_iou_assgin() -> None:
    a = IoUAssign(threshold=0.4)
    inputs = torch.zeros(3, 4)
    inputs[0] = torch.tensor([0, 0, 5, 5])
    inputs[1] = torch.tensor([10, 10, 20, 20])
    inputs[2] = torch.tensor([5, 10, 10, 15])

    targets = torch.zeros(2, 4)
    targets[0] = torch.tensor([0, 0, 3, 3])
    targets[1] = torch.tensor([5, 10, 10, 20])
    pair = a(inputs, targets)
    assert len(pair) == 1
    assert pair[0].tolist() == [2, 1]


@pytest.mark.parametrize("in_len, tgt_len", [(0, 1), (1, 0), (0, 0)])
def test_iou_assgin_empty(in_len: int, tgt_len: int) -> None:
    a = IoUAssign(threshold=0.4)
    inputs = torch.zeros(in_len, 4)
    targets = torch.zeros(tgt_len, 4)
    pair = a(inputs, targets)
    assert len(pair) == 0


def test_atss() -> None:
    inputs = torch.zeros(5, 4)
    inputs[0] = torch.tensor([0, 0, 5, 5])
    inputs[1] = torch.tensor([0, 0, 3, 3])
    inputs[2] = torch.tensor([0, 0, 2, 2])
    inputs[3] = torch.tensor([10, 10, 20, 20])
    inputs[4] = torch.tensor([5, 10, 10, 15])

    targets = torch.zeros(3, 4)
    targets[0] = torch.tensor([0, 0, 4, 4])
    targets[1] = torch.tensor([5, 10, 10, 20])
    targets[2] = torch.tensor([10, 10, 20, 20])
    a = ATSS(topk=6)
    pair = a(inputs, targets)


def test_simota() -> None:
    anchor_points = torch.zeros(4, 2)
    anchor_points[0] = torch.tensor([5, 5])
    anchor_points[1] = torch.tensor([5, 5])
    anchor_points[2] = torch.tensor([5, 5])
    anchor_points[3] = torch.tensor([5, 5])

    pred_boxes = torch.zeros(4, 4)
    pred_boxes[0] = torch.tensor([5, 5, 8, 8])
    pred_boxes[1] = torch.tensor([5, 5, 8, 8])
    pred_boxes[2] = torch.tensor([5, 5, 8, 8])
    pred_boxes[3] = torch.tensor([5, 5, 8, 8])
    pred_scores = torch.tensor([0.8, 0.4, 0.8, 0.8])
    strides = torch.tensor([1.0, 2.0, 4.0, 8.0])

    gt_boxes = torch.zeros(1, 4)
    gt_boxes[0] = torch.tensor([3, 3, 9, 9])
    a = SimOTA(topk=6, radius=0.5)
    pair = a(anchor_points, pred_boxes, pred_scores, gt_boxes, strides)
    assert pair.tolist() == [
        [0, 2],
    ]
