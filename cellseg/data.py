import numpy as np
import torch
from torch import Tensor
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks, save_image
import pandas as pd


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
    mask = mask.reshape(shape[0], shape[1]).astype(bool)
    return torch.from_numpy(mask)


def get_masks(df: pd.DataFrame, image_id: str) -> Tensor:
    rows = df[df["id"] == image_id].apply(
        lambda r: decode_rle_mask(r.annotation, (r.height, r.width)), axis=1
    )
    masks = torch.stack(rows.tolist())
    return masks


def draw_save(image: Tensor, masks: Tensor, path: str) -> None:
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    plot = draw_segmentation_masks(image, masks, alpha=0.3)
    save_image(plot / 255, path)
