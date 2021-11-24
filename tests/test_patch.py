import torch
import torch.nn as nn
from cellseg.util import grid, draw_save
from torchvision.utils import make_grid
from cellseg.data import (
    CellTrainDataset,
    Tranform,
)
from cellseg.util import ToPatches


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
    images, patch_batch = to_patches(image)

    for i, (image, patches) in enumerate(zip(images, patch_batch)):
        assert patches.shape == (20, 3, patch_size, patch_size)
        print(patches.shape)
        plot = make_grid(patches, nrow=w // patch_size)
        draw_save(f"test_outputs/test-patch-{i}-padded.png", image)
        draw_save(f"test_outputs/test-patch-{i}-patches.png", plot)
