import pytest
import torch
from cellseg.backbones import EfficientNetFPN, efficientnet_channels, CSPDarknet


@pytest.mark.parametrize("name", list(efficientnet_channels.keys()))
def test_efficient_net_fpn(name: str) -> None:
    p1_size = 512
    image = torch.rand(1, 3, p1_size, p1_size)
    backbone = EfficientNetFPN(name)
    features = backbone(image)
    expand_len = 7
    expected_sizes = [p1_size // s for s in backbone.strides]
    assert (
        len(features)
        == expand_len
        == len(backbone.out_channels)
        == len(backbone.strides)
    )
    for f, s, c in zip(features, expected_sizes, backbone.out_channels):
        assert f.size(1) == c
        assert f.shape[2] == s
        assert f.shape[3] == s


def test_darknet() -> None:
    image_size = 512
    image = torch.rand(1, 3, image_size, image_size)
    backbone = CSPDarknet()
    features = backbone(image)
