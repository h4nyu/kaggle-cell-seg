import torch
from cellseg.coord_conv import CoordConv


def test_coord_conv() -> None:
    m = CoordConv()
    feat = torch.rand(1, 1, 3, 4)  # (B, C, H, W)
    res = m(feat)
    assert res.shape == (1, 3, 3, 4)
    assert res[0, 1, 0, 0] == -1
    assert res[0, 2, 0, 0] == -1
    assert res[0, 1, 0, -1] == 1
    assert res[0, 2, 0, -1] == -1
