import torch
from cellseg.center_mask import CenterMask
from cellseg.backbones import EfficientNetFPN


def test_model() -> None:
    num_classes = 2
    mask_size = 16
    images = torch.rand(1, 3, 512, 512)
    backbone = EfficientNetFPN("efficientnet-b2")
    category_feat_range = (4, 6)
    num_classes = 2

    model = CenterMask(
        hidden_channels=64,
        backbone=backbone,
        mask_size=mask_size,
        num_classes=num_classes,
        category_feat_range=category_feat_range,
    )
    category_grids, size_grids, offset_grids, mask_grids, sliency_masks = model(images)
    assert category_grids.shape == (1, num_classes, 32, 32)
    assert size_grids.shape == (1, 2, 32, 32)
    assert offset_grids.shape == (1, 2, 32, 32)
    assert mask_grids.shape == (1, mask_size ** 2, 32, 32)
    assert sliency_masks.shape == (1, 1, 512, 512)
    # assert all_masks.shape == (1, grid_size ** 2, *image_batch.shape[2:])
