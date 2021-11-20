import pytest
import itertools
from cellseg.data import (
    decode_rle_mask,
    get_masks,
    draw_save,
    CellTrainDataset,
    Tranform,
    inv_normalize,
)
from hydra import compose, initialize
import pandas as pd
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt
from pathlib import Path

initialize(config_path="../config")
cfg = compose(config_name="config")

has_data = Path(cfg.data.train_file_path).exists()
if has_data:
    train_df = pd.read_csv(cfg.data.train_file_path)
    first_row = train_df.loc[0]


@pytest.mark.skipif(not has_data, reason="no data volume")
def test_decode_rle_mask_and_view() -> None:
    shape = (first_row.height, first_row.width)
    rle_mask = first_row.annotation
    masks = decode_rle_mask(rle_mask, shape)

    assert masks.shape == shape


@pytest.mark.skipif(not has_data, reason="no data volume")
@pytest.mark.parametrize(
    "image_id, cell_type",
    list(
        itertools.product(
            ["0a6ecc5fe78a", "0ba181d412da", "0030fd0e6378"],
            [c for c in cfg.data.cell_types],
        )
    ),
)
def test_get_masks_and_plot(image_id: str, cell_type: str) -> None:
    masks = get_masks(train_df, image_id, cell_type)
    if masks is not None:
        assert masks.shape[0] == sum(
            (train_df["id"] == image_id) & (train_df["cell_type"] == cell_type)
        )
        img = read_image(os.path.join(cfg.data.train_images_path, f"{image_id}.png"))
        draw_save(
            os.path.join(cfg.data.root_path, f"test-plot-{image_id}-{cell_type}.png"),
            img / 255,
            masks=masks,
        )


@pytest.mark.skipif(not has_data, reason="no data volume")
def test_cell_dataset() -> None:
    transform = Tranform(original_size=cfg.original_size)
    dataset = CellTrainDataset(
        transform=transform,
    )
    assert len(dataset) == 606
    sample = dataset[0]
    assert sample is not None
    image = sample["image"]
    masks = sample["masks"]
    labels = sample["labels"]
    draw_save("/store/test-dataset.png", image, masks)
    assert image.shape == (3, cfg.original_size, cfg.original_size)
    assert image.shape[1:] == masks.shape[1:]
    assert labels.shape[0] == masks.shape[0]
