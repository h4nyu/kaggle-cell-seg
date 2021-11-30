import torch
from cellseg.blocks import ConvBnAct


def test_conv_norm_act() -> None:
    in_channels = 10
    out_channels = 3
    conv = ConvBnAct(
        in_channels=in_channels,
        out_channels=out_channels,
    )
    feat = torch.rand(1, in_channels, 3, 4)
    out = conv(feat)
    assert out.shape == (1, out_channels, 3, 4)


