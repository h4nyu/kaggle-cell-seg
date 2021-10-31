import numpy as np
import torch
from torch import Tensor


def seed(num: int) -> None:
    torch.manual_seed(num)
    np.random.seed(num)


def decode_rle_mask(rle_mask: str, shape: tuple[int, int]) -> Tensor:
    mask_nums = rle_mask.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (mask_nums[0:][::2], mask_nums[1:][::2])
    ]
    ends = starts + lengths
    mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    mask = mask.reshape(shape[0], shape[1])
    return torch.from_numpy(mask)
