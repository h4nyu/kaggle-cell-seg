import pytest
import torch
from cellseg.solo.heads import Head


@pytest.mark.parametrize(
    "channels, reductions",
    [
        ([12, 24, 48], [1, 2, 4]),
        ([12, 24, 48, 64], [1, 1, 1, 4]),
        ([12, 24, 48, 64], [1, 2, 2, 4]),
    ],
)
def test_mask_head(channels: list[int], reductions: list[int]) -> None:
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
    )
    res = head(features)
    assert res.shape[2:] == features[0].shape[2:]
    assert res.shape[:2] == (1, num_classes)
