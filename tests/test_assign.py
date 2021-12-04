import pytest
import torch
from cellseg.assign import IoUAssign, ATSS


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

    targets = torch.zeros(2, 4)
    targets[0] = torch.tensor([0, 0, 4, 4])
    targets[1] = torch.tensor([5, 10, 10, 20])
    a = ATSS(topk=5)
    pair = a(inputs, targets)
