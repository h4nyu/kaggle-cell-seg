import torch
from cellseg.solo.heads import Head


def test_mask_head() -> None:
    in_channels = 64
    out_channels = 64
    base_resolution = 32
    num_classes = 64 * 64
    fpn_length = 5

    features = [
        torch.rand(1, in_channels, base_resolution * 2 ** i, base_resolution * 2 ** i)
        for i in reversed(range(fpn_length))
    ]
    head = Head(
        in_channels=in_channels,
        out_channels=out_channels,
        num_classes=num_classes,
        fpn_length=fpn_length,
    )
    res = head(features)
    assert res.shape[2:] == features[0].shape[2:]
    assert res.shape[:2] == (1, num_classes)
