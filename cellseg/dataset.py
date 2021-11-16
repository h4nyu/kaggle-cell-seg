import torch
from torch.utils.data import Dataset
from torch import Tensor
from cellseg.data import get_masks
import pandas as pd
from torchvision.io import read_image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from typing import TypedDict, Optional, cast
import numpy as np

TrainItem = TypedDict(
    "TrainItem",
    {
        "id": str,
        "image": Tensor,
        "masks": Tensor,
        "labels": Tensor,
    },
)


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


def get_fold_indices(
    dataset: Dataset, n_splits: int = 5, index: int = 0, seed: int = 0
) -> tuple[list[int], list[int]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x = np.arange(len(dataset))
    y = dataset.stratums
    return list(splitter.split(x, y))[index]


class CellTrainDataset(Dataset):
    def __init__(
        self, img_dir: str = "/store/train", train_csv: str = "/store/train.csv"
    ) -> None:
        self.img_dir = img_dir
        df = pd.read_csv(train_csv)
        self.df = df
        self.indecies = self.df["id"].unique()
        self.stratums = LabelEncoder().fit_transform(
            [df[df["id"] == id].iloc[0]["cell_type"] for id in self.indecies]
        )

    def __len__(self) -> int:
        return len(self.indecies)

    def __getitem__(self, idx: int) -> Optional[TrainItem]:
        image_id: str = self.indecies[idx]
        masks = get_masks(df=self.df, image_id=image_id)
        if masks is None:
            return None
        labels = torch.zeros(masks.shape[0])
        image = read_image(f"{self.img_dir}/{image_id}.png")
        return dict(
            id=image_id,
            image=image,
            masks=masks,
            labels=labels,
        )
