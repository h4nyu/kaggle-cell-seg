import torch
from cellseg.backbones import EfficientNetFPN


def test_efficient_net_fpn() -> None:
    p1_size = 512
    image = torch.rand(1, 3, p1_size, p1_size)
    backbone = EfficientNetFPN()
    features = backbone(image)
    expand_len = 6
    expected_sizes = [p1_size / (2 ** i) for i in range(expand_len)]
    assert len(features) == expand_len
    for f, s in zip(features, expected_sizes):
        assert f.shape[2] == s
        assert f.shape[3] == s
