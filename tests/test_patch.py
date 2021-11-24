import torch
import torch.nn as nn
from cellseg.util import grid, draw_save
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
    draw_save("test_outputs/test-patch-base.png", image[0])
    patch_size = 128
    pad_size = (
            (h % patch_size) // 4,
            (w % patch_size) // 4,
            (h % patch_size) // 4,
            (w % patch_size) // 4,
    )
    pad = nn.ZeroPad2d(pad_size)
    image = pad(image)

    patches = image.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    patches = (
        patches.permute([0, 2, 3, 1, 4, 5])
        .contiguous()
        .view(-1, 3, patch_size, patch_size)
    )
    # assert patches.shape == (6, 3, patch_size, patch_size)
    print(patches.shape)
    plot = make_grid(patches, nrow=w // patch_size)
    draw_save("test_outputs/test-patch.png", plot)
