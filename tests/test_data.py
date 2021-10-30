from cellseg.data import decode_rle_mask
from cellseg.config import TRAIN_FILE_PATH
import pandas as pd

train_df = pd.read_csv(TRAIN_FILE_PATH)


def test_decode_rle_mask() -> None:
    row = train_df.loc[0]
    shape = (row.height, row.width)
    rle_mask = row.annotation
    mask = decode_rle_mask(rle_mask, shape)
    assert mask.shape == shape
