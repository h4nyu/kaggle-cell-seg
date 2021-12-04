import pytest
import torch
from typing import Optional
from cellseg.heads import Head, MaskHead, CSPUpHead


@pytest.mark.parametrize(
    "in_channels, reductions, coord_level",
    [
        ([12, 24, 48], [1, 2, 4], None),
        ([12, 24, 48], [1, 2, 4], 0),
        ([12, 24, 48], [1, 2, 4], 1),
        ([12, 24, 48, 64], [1, 1, 1, 2], 0),
        ([12, 24, 48, 64], [1, 1, 1, 2], 1),
        ([12, 24, 48, 64], [1, 1, 1, 2], 2),
    ],
)
def test_solo_head(
    in_channels: list[int], reductions: list[int], coord_level: Optional[int]
) -> None:
    hidden_channels = 64
    base_resolution = 512
    out_channels = 3 * 3

    features = [
        torch.rand(1, c, base_resolution // s, base_resolution // s)
        for (c, s) in zip(in_channels, reductions)
    ]
    head = Head(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        in_channels=in_channels,
        reductions=reductions,
        coord_level=coord_level,
    )
    res = head(features)
    assert res.shape[2:] == features[0].shape[2:]
    assert res.shape[:2] == (1, out_channels)


@pytest.mark.parametrize(
    "in_channels, reductions, use_cord",
    [
        ([12, 24, 48], [1, 2, 4], True),
        ([12, 24, 48], [1, 2, 4], False),
        ([12, 24, 48, 64], [1, 1, 1, 2], True),
        ([12, 24, 48, 64], [1, 1, 1, 2], False),
        ([12, 24, 48, 64], [1, 2, 2, 4], True),
    ],
)
def test_csp_up_head(
    in_channels: list[int], reductions: list[int], use_cord: bool
) -> None:
    out_channels = 16
    hidden_channels = 64
    size = 128
    inputs = [
        torch.rand(2, c, size // r, size // r) for c, r in zip(in_channels, reductions)
    ]
    head = CSPUpHead(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        reductions=reductions,
        use_cord=use_cord,
        depth=2,
    )
    outs = head(inputs)
    assert outs.shape == (2, out_channels, size, size)


def test_mask_head() -> None:
    in_channels = 32
    out_channels = 16
    size = 128
    inputs = torch.rand(2, in_channels, size, size)
    head = MaskHead(in_channels=in_channels, out_channels=out_channels, depth=2)
    outs = head(inputs)
    assert outs.shape == (2, out_channels, size, size)
