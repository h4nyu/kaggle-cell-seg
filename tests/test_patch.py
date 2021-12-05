import torch
import torch.nn as nn
from cellseg.utils import ToPatches, MergePatchedMasks, grid, draw_save
from torchvision.utils import make_grid
from cellseg.data import (
    CellTrainDataset,
    Tranform,
)


def test_patch() -> None:
    dataset = CellTrainDataset(
        img_dir="data",
        train_csv="data/annotation.csv",
    )
    sample = dataset[0]
    assert sample is not None
    image = sample["image"]
    masks = sample["masks"]
    labels = sample["labels"]

    _, h, w = image.shape
    image = image.view(1, *image.shape)
    patch_size = 128
    to_patches = ToPatches(patch_size=patch_size)
    images, patch_batch, patch_grid = to_patches(image)
    assert patch_grid[1].tolist() == [128, 0]
    assert patch_grid[6].tolist() == [0, 128]

    for i, (image, patches) in enumerate(zip(images, patch_batch)):
        assert patches.shape == (len(patch_grid), 3, patch_size, patch_size)
        patches[1, :, 10:50, 50:60] = 1
        plot = make_grid(patches, nrow=image.shape[2] // patch_size, padding=1)
        draw_save(f"test_outputs/test-patch-{i}-padded.png", image)
        draw_save(f"test_outputs/test-patch-{i}-patches.png", plot)


def test_merge_masks() -> None:
    merge = MergePatchedMasks(patch_size=10)
    # 2 x 1 grid
    patch_grid = torch.tensor(
        [
            [
                0,
                0,
            ],
            [
                10,
                0,
            ],
        ]
    )
    mask0 = torch.zeros(10, 10)
    mask0[4, 4] = 1
    mask1 = torch.zeros(10, 10)
    mask1[5, 6] = 1

    mask2 = torch.zeros(10, 10)
    mask2[1, 1] = 1
    batch = [torch.stack([mask0, mask1]), torch.stack([mask2])]
    merged_masks = merge(batch, patch_grid)
    assert len(merged_masks) == 3
    assert merged_masks[0, 4, 4] == 1
    assert merged_masks[1, 5, 6] == 1
    assert merged_masks[2, 1, 11] == 1
