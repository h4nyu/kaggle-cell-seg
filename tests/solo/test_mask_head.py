import torch
from cellseg.solo.mask_head import MaskHead


def test_mask_head() -> None:
    in_channels = 3
    out_channels = 4
    base_resolution = 4
    num_classes = 10
    features = [
        torch.rand(1, in_channels, base_resolution * 2 ** i, base_resolution * 2 ** i)
        for i in reversed(range(3))
    ]
    head = MaskHead(
        in_channels=in_channels,
        out_channels=out_channels,
        num_classes=num_classes,
        fpn_length=len(features),
    )
    res = head(features)
    assert res.shape[2:] == features[0].shape[2:]
    assert res.shape[:2] == (1, num_classes)
