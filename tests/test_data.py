from cellseg.data import decode_rle_mask, get_masks
from cellseg.config import TRAIN_FILE_PATH, TRAIN_PATH, ROOT_PATH
import pandas as pd
from torchvision.utils import draw_segmentation_masks, save_image
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


def test_get_masks() -> None:
    masks = get_masks(train_df, first_row.id)
    assert masks.shape == (
        sum(train_df["id"] == first_row.id),
        first_row.height,
        first_row.width,
    )
