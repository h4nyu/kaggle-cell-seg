import numpy as np
from torch import Tensor


def decode_rle_mask(rle_mask: str, shape: tuple[float, float]) -> Tensor:
    print("aaa")
