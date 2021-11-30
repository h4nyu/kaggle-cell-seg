import torch
from cellseg.modules import MaskHead


def test_mask_head() -> None:
    in_channels = 32
    out_channels = 16
    size = 128
    inputs = torch.rand(2, in_channels, size, size)
    head = MaskHead(in_channels=in_channels, out_channels=out_channels, depth=2)
    outs = head(inputs)
    assert outs.shape == (2, out_channels, size * 2, size * 2)
