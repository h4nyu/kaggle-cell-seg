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
import albumentations as A


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


TrainItem = TypedDict(
    "TrainItem",
    {
        "id": str,
        "image": Tensor,
        "masks": Tensor,
        "labels": Tensor,
    },
)

normalize_mean = [0.485, 0.456, 0.406]
normalize_std = [0.229, 0.224, 0.225]
inv_normalize = A.Normalize(
    mean=[-m / s for m, s in zip(normalize_mean, normalize_std)],
    std=[1 / s for s in normalize_std],
)


class TrainTranform:
    def __init__(self, original_size: int):

        self.transform = A.Compose(
            [
                A.Flip(),
                A.RandomRotate90(),
                # A.RandomScale(scale_limit=(0.95, 1.05)),
                A.RandomCrop(width=original_size, height=original_size, p=1.0),
                ToTensorV2(),
            ]
        )

    def __call__(self, *args: Any, **kargs: Any) -> Any:
        return self.transform(*args, **kargs)


class Tranform:
    def __init__(self, original_size: int):

        self.transform = A.Compose(
            [
                A.RandomCrop(width=original_size, height=original_size, p=1.0),
                ToTensorV2(),
            ]
        )

    def __call__(self, *args: Any, **kargs: Any) -> Any:
        return self.transform(*args, **kargs)


def collate_fn(batch: list[TrainItem]) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    images: list[Tensor] = []
    mask_batch: list[Tensor] = []
    label_batch: list[Tensor] = []
    for row in batch:
        images.append(row["image"])
        mask_batch.append(row["masks"])
        label_batch.append(row["labels"])
    return (
        torch.stack(images),
        mask_batch,
        label_batch,
    )


class CellTrainDataset(Dataset):
    def __init__(
        self,
        img_dir: str = "/store/train",
        train_csv: str = "/store/train.csv",
        transform: Optional[Callable] = None,
    ) -> None:
        self.img_dir = img_dir
        df = pd.read_csv(train_csv)
        self.df = df
        self.indecies = self.df["id"].unique()
        self.stratums = LabelEncoder().fit_transform(
            [df[df["id"] == id].iloc[0]["cell_type"] for id in self.indecies]
        )
        self.transform = ToTensorV2() if transform is None else transform

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
        empty_filter = masks.sum(dim=[1, 2]) > 0
        masks = masks[empty_filter]
        labels = torch.from_numpy(transformed["labels"])
        labels = labels[empty_filter]
        return dict(
            id=image_id,
            image=image,
            masks=masks,
            labels=labels,
        )


def get_fold_indices(
    dataset: CellTrainDataset, n_splits: int = 5, index: int = 0, seed: int = 0
) -> tuple[list[int], list[int]]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x = np.arange(len(dataset))
    y = dataset.stratums
    return list(splitter.split(x, y))[index]
