import pytest
import torch
from cellseg.heads import Head, MaskHead, CSPUpHead


@pytest.mark.parametrize(
    "channels, reductions",
    [
        ([12, 24, 48], [1, 2, 4]),
        # ([12, 24, 48, 64], [1, 1, 1, 2]),
        # ([12, 24, 48, 64], [1, 2, 2, 4]),
    ],
)
def test_solo_head(channels: list[int], reductions: list[int]) -> None:
    in_channels = 64
    hidden_channels = 64
    base_resolution = 512
    num_classes = 3 * 3

    features = [
        torch.rand(1, c, base_resolution // s, base_resolution // s)
        for (c, s) in zip(channels, reductions)
    ]
    head = Head(
        hidden_channels=hidden_channels,
        num_classes=num_classes,
        channels=channels,
        reductions=reductions,
        use_cord=True,
    )
    res = head(features)
    assert res.shape[2:] == features[0].shape[2:]
    assert res.shape[:2] == (1, num_classes)


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
    assert outs.shape == (2, out_channels, size * 2, size * 2)
