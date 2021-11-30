import torch
from cellseg.blocks import (
    ConvBnAct,
    SpatialAttention,
    SpatialPyramidPooling,
    ReversedCSP,
    CSPUpBlock,
)


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


def test_spatial_attention() -> None:
    in_channels = 10
    sp = SpatialAttention(
        in_channels=in_channels,
    )
    inputs = torch.rand(1, in_channels, 128, 128)
    res = sp(inputs)
    assert res.shape == inputs.shape


def test_spatial_pyramid_pooling() -> None:
    inputs = torch.rand(1, 32, 128, 128)
    kernel_sizees = [5, 9, 13]
    spp = SpatialPyramidPooling()
    res = spp(inputs)
    assert res.shape == (1, 32 * (len(kernel_sizees) + 1), 128, 128)


def test_reversed_scp() -> None:
    in_channels = 16
    out_channels = 32
    depth = 4
    rcsp = ReversedCSP(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
    )

    inputs = torch.rand(1, in_channels, 128, 128)
    res = rcsp(inputs)
    assert res.shape == (1, out_channels, 128, 128)


def test_csp_up_block() -> None:
    in_channels = (16, 32)
    out_channels = 32
    depth = 4
    up = CSPUpBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        depth=depth,
    )

    inputs = (
        torch.rand(1, in_channels[0], 128, 128),
        torch.rand(1, in_channels[1], 128 * 2, 128 * 2),
    )
    res = up(*inputs)
    assert res.shape == (1, out_channels, 128 * 2, 128 * 2)
