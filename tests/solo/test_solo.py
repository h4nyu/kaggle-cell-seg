import torch
from cellseg.solo import Solo, Criterion
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
        hidden_channels=64,
        grid_size=grid_size,
        category_feat_range=category_feat_range,
        mask_feat_range=mask_feat_range,
    )
    category_grid, all_masks = solo(image_batch)
    assert category_grid.shape == (1, num_classes, grid_size, grid_size)
    assert all_masks.shape == (1, grid_size ** 2, *image_batch.shape[2:])


def test_loss() -> None:
    batch_size = 1
    num_classes = 2
    grid_size = 10
    original_size = 100

    # net outputss
    pred_category_grids = (
        torch.ones(batch_size, num_classes, grid_size, grid_size) * -100
    )
    all_masks = (
        torch.ones(batch_size, grid_size * grid_size, original_size, original_size)
        * -100
    )

    # data adaptor outputs
    gt_category_grids = torch.zeros(batch_size, num_classes, grid_size, grid_size)
    gt_mask_batch = [torch.zeros(3, original_size, original_size)]
    mask_index_batch = [torch.tensor([1, 2, 3])]  # same len to mask_batch item

    loss = Criterion()
    loss_value = loss(
        inputs=(pred_category_grids, all_masks),
        targets=(gt_category_grids, gt_mask_batch, mask_index_batch),
    )
    assert loss_value == 0
