import torch
from cellseg.solo import Solo, Loss
from cellseg.backbones import EfficientNetFPN


def test_solo() -> None:
    image_batch = torch.rand(1, 3, 512, 512)
    backbone = EfficientNetFPN("efficientnet-b2")
    category_feat_range = (4, 6)
    mask_feat_range = (0, 4)
    num_classes = 2
    grid_size = 32

    solo = Solo(
        num_classes=num_classes,
        backbone=backbone,
        out_channels=64,
        grid_size=grid_size,
        category_feat_range=category_feat_range,
        mask_feat_range=mask_feat_range,
    )
    category_grid, all_masks = solo(image_batch)
    assert category_grid.shape == (1, num_classes, grid_size, grid_size)
    assert all_masks.shape == (1, grid_size ** 2, *image_batch.shape[2:])


def test_solo() -> None:
    ...
    # image_batch = torch.rand(1, 3, 512, 512)
    # backbone = EfficientNetFPN("efficientnet-b2")
    # category_feat_range = (4, 6)
    # mask_feat_range = (0, 4)
    # num_classes = 2
    # grid_size = 32

    # solo = Solo(
    #     num_classes=num_classes,
    #     backbone=backbone,
    #     out_channels=64,
    #     grid_size=grid_size,
    #     category_feat_range=category_feat_range,
    #     mask_feat_range=mask_feat_range,
    # )
    # category_grid, all_masks = solo(image_batch)
    # assert category_grid.shape == (1, num_classes, grid_size, grid_size)
    # assert all_masks.shape == (1, grid_size ** 2, *image_batch.shape[2:])
