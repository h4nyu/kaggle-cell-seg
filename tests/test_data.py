import pytest
import itertools
from cellseg.data import decode_rle_mask, get_masks, draw_save
from cellseg.config import train_file_path
import pandas as pd
from torchvision.io import read_image, CellType, root_path
import os
import matplotlib.pyplot as plt

train_df = pd.read_csv(train_file_path)
first_row = train_df.loc[0]


def test_decode_rle_mask_and_view() -> None:
    shape = (first_row.height, first_row.width)
    rle_mask = first_row.annotation
    masks = decode_rle_mask(rle_mask, shape)

    assert masks.shape == shape


@pytest.mark.parametrize(
    "image_id, cell_type",
    list(
        itertools.product(
            ["0a6ecc5fe78a", "0ba181d412da", "0030fd0e6378"],
            [e.value for e in CellType],
        )
    ),
)
def test_get_masks_and_plot(image_id: str, cell_type: CellType) -> None:
    masks = get_masks(train_df, image_id, cell_type)
    if masks is not None:
        assert masks.shape[0] == sum(
            (train_df["id"] == image_id) & (train_df["cell_type"] == cell_type)
        )
        img = read_image(os.path.join(root_path, "train", f"{image_id}.png"))
        draw_save(
            img / 255,
            os.path.join(root_path, f"test-plot-{image_id}-{cell_type}.png"),
            masks=masks,
        )
