import pytest
from cellseg.data import decode_rle_mask, get_masks, draw_save
from cellseg.config import TRAIN_FILE_PATH, TRAIN_PATH, ROOT_PATH
import pandas as pd
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt

train_df = pd.read_csv(TRAIN_FILE_PATH)
first_row = train_df.loc[0]


def test_decode_rle_mask_and_view() -> None:
    shape = (first_row.height, first_row.width)
    rle_mask = first_row.annotation
    masks = decode_rle_mask(rle_mask, shape)

    assert masks.shape == shape


@pytest.mark.parametrize("image_id", ["0a6ecc5fe78a", "0ba181d412da", "0030fd0e6378"])
def test_get_masks_and_plot(image_id: str) -> None:
    masks = get_masks(train_df, image_id)
    assert masks.shape[0] == sum(train_df["id"] == image_id)
    img = read_image(os.path.join(ROOT_PATH, "train", f"{image_id}.png"))
    draw_save(img, masks, os.path.join(ROOT_PATH, f"test-plot-{image_id}.png"))
