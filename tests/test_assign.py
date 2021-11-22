import torch
from cellseg.assign import IoUAssign
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
