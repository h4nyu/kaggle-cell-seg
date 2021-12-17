from typing import Optional
import numpy as np
import torch
from torch import Tensor

import torchvision.transforms.functional as F
import pandas as pd
from typing import TypedDict, Optional, cast, Callable, Any, Union
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torchvision.ops import masks_to_boxes
import albumentations as A

CELL_TYPES = ["astro", "cort", "shsy5y"]

def seed(num: int) -> None:
    torch.manual_seed(num)
    np.random.seed(num)


def decode_rle_mask(rle_mask: str, shape: tuple[int, int]) -> Tensor:
    mask_nums = rle_mask.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (mask_nums[0:][::2], mask_nums[1:][::2])
    ]
    ends = (starts - 1) + lengths  # start is 1-based indexing
    mask = np.zeros((shape[0] * shape[1]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    mask = mask.reshape(shape[0], shape[1]).astype(bool)
    return torch.from_numpy(mask)


def get_masks(
    df: pd.DataFrame, image_id: str, cell_type: Optional[str] = None
) -> Optional[Tensor]:
    rows = df[df["id"] == image_id]
    if cell_type is not None:
        rows = rows[rows["cell_type"] == cell_type]
    if len(rows) == 0:
        return None
    rows = rows.apply(
        lambda r: decode_rle_mask(r.annotation, (r.height, r.width)), axis=1
    )
    masks = torch.stack(rows.tolist())
    return masks


TrainBatch = TypedDict(
    "TrainBatch",
    {
        "images": Tensor,
        "mask_batch": list[Tensor],
        "box_batch": list[Tensor],
        "label_batch": list[Tensor],
    },
)

TrainItem = TypedDict(
    "TrainItem",
    {
        "id": str,
        "image": Tensor,
        "masks": Tensor,
        "boxes": Tensor,
        "labels": Tensor,
    },
)

normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
inv_normalize = A.Normalize(
    mean=[-m / s for m, s in zip(normalize_mean, normalize_std)],
    std=[1 / s for s in normalize_std],
)

PadResize = lambda size: A.Compose(
    [
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(
            min_height=size,
            min_width=size,
            border_mode=0,
        ),
    ]
)


class TrainTranform:
    def __init__(self, size: int, use_patch: bool = False):

        self.transform = A.Compose(
            [
                A.RandomBrightness(),
                A.Flip(),
                A.RandomRotate90(),
                A.ShiftScaleRotate(scale_limit=(0.2, 0.2), p=1.0, border_mode=0),
                A.RandomCrop(width=size, height=size, p=1.0)
                if use_patch
                else A.Resize(width=size, height=size, p=1.0),
                # else PadResize(size=size),
                ToTensorV2(),
            ]
        )

    def __call__(self, *args: Any, **kargs: Any) -> Any:
        return self.transform(*args, **kargs)


class Tranform:
    def __init__(self, size: int, use_patch: bool = False):

        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size=size),
                A.RandomCrop(width=size, height=size, p=1.0)
                if use_patch
                else A.Resize(width=size, height=size, p=1.0),
                # else PadResize(size=size),
                ToTensorV2(),
            ]
        )

    def __call__(self, *args: Any, **kargs: Any) -> Any:
        return self.transform(*args, **kargs)


class CollateFn:
    def __init__(self, transform: Any = None) -> None:
        self.transform = transform

    def __call__(self, batch: list[TrainItem]) -> TrainBatch:
        images: list[Tensor] = []
        mask_batch: list[Tensor] = []
        box_batch: list[Tensor] = []
        label_batch: list[Tensor] = []
        for row in batch:
            images.append(row["image"])
            mask_batch.append(row["masks"])
            box_batch.append(row["boxes"])
            label_batch.append(row["labels"])

        res = dict(
            images=torch.stack(images),
            mask_batch=mask_batch,
            box_batch=box_batch,
            label_batch=label_batch,
        )
        if self.transform is not None:
            return self.transform(res)
        return res


class CellTrainDataset(Dataset):
    def __init__(
        self,
        img_dir: str = "/store/train",
        train_csv: str = "/store/train.csv",
        transform: Optional[Callable] = None,
        smallest_area: int = 0,
    ) -> None:
        self.img_dir = img_dir
        df = pd.read_csv(train_csv)
        self.df = df
        self.indecies = self.df["id"].unique()
        self.stratums = df.groupby("id").first()["cell_type"].values
        self.transform = ToTensorV2() if transform is None else transform
        self.smallest_area = smallest_area

    def __len__(self) -> int:
        return len(self.indecies)

    def __getitem__(self, idx: int) -> Optional[TrainItem]:
        image_id: str = self.indecies[idx]
        masks = get_masks(df=self.df, image_id=image_id)
        if masks is None:
            return None
        labels = torch.zeros(masks.shape[0])
        image = cv2.imread(f"{self.img_dir}/{image_id}.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transform(
            image=image,
            masks=[m.numpy() for m in masks.short().unbind()],
            labels=labels.numpy(),
        )
        image = transformed["image"] / 255
        masks = torch.stack([torch.from_numpy(m) for m in transformed["masks"]]).bool()
        empty_filter = masks.sum(dim=[1, 2]) > self.smallest_area
        masks = masks[empty_filter]
        boxes = masks_to_boxes(masks)
        labels = torch.from_numpy(transformed["labels"]).long()
        labels = labels[empty_filter]
        return dict(
            id=image_id,
            image=image,
            masks=masks,
            labels=labels,
            boxes=boxes,
        )


def get_fold_indices(
    dataset: CellTrainDataset, n_splits: int = 5, index: int = 0, seed: int = 0
) -> tuple[list[int], list[int]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x = np.arange(len(dataset))
    y = dataset.stratums
    print(y)
    return list(splitter.split(x, y))[index]
