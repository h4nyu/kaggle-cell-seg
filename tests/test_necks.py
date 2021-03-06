import torch
from cellseg.necks import CSPNeck


def test_csp_neck() -> None:
    in_channels = [256, 512, 512, 1024]
    out_channels = [256, 512, 512, 1024]
    size = 256
    strides = [1, 2, 2, 4]

    features = [
        torch.rand(1, c, size // r, size // r) for (c, r) in zip(in_channels, strides)
    ]
    neck = CSPNeck(in_channels=in_channels, out_channels=out_channels, strides=strides)
    out_features = neck(features)
    for f, c, r in zip(out_features, out_channels, strides):
        assert f.shape == (1, c, size // r, size // r)
