from cellseg.data import decode_rle_mask
from cellseg.config import TRAIN_FILE_PATH
import pandas as pd

train_df = pd.read_csv(TRAIN_FILE_PATH)

print(train_df)

def test_decode_rle_mask() -> None:
    decode_rle_mask("", (120.0, 10.0))
