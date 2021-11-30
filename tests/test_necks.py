import torch
from cellseg.necks import CSPNeck


def test_csp_neck() -> None:
    in_channels = [256, 512, 1024]
    out_channels = [256, 512, 1024]
    size = 512
    reductions = [1, 2, 4]

    features = [
        torch.rand(1, c, size // r, size // r)
        for (c, r) in zip(in_channels, reductions)
    ]
    neck = CSPNeck(in_channels=in_channels, out_channels=out_channels)
