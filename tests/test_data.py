from cellseg.data import decode_rle_mask
from cellseg.config import TRAIN_FILE_PATH, TRAIN_PATH
import pandas as pd
from torchvision.utils import draw_segmentation_masks
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt

train_df = pd.read_csv(TRAIN_FILE_PATH)
first_row = train_df.loc[0]


def test_decode_rle_mask_and_view() -> None:
    shape = (first_row.height, first_row.width)
    rle_mask = first_row.annotation
    mask = decode_rle_mask(rle_mask, shape)

    assert mask.shape == shape

    image_path = os.join(TRAIN_PATH, f"{first_row.id}.png")
    img = read_image(image_path)
    draw_segmentation_masks(img, mask=mask, alpa=0.7)
