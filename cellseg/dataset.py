import torch
from torch.utils.data import Dataset, Subset
from torch import Tensor
from cellseg.data import get_masks
import pandas as pd
from torchvision.io import read_image
from sklearn.model_selection import StratifiedGroupKFold
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


def get_fold_indices(
    dataset: Dataset, n_splits: int = 5, fold: int = 0, seed: int = 0
) -> tuple[list[int], list[int]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x = np.arange(len(dataset))
    y = dataset.stratums
    groups = dataset.groups
    return list(splitter.split(x, y, groups))[fold]


class CellTrainDataset(Dataset):
    def __init__(
        self, img_dir: str = "/store/train", train_csv: str = "/store/train.csv"
    ) -> None:
        self.img_dir = img_dir
        self.df = pd.read_csv(train_csv)
        self.indecies = self.df["id"].unique()

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


class TrainSet(Subset):
    def __init__(
        self, dataset: Dataset, n_splits: int = 5, fold: int = 0, seed: int = 0
    ):
        indices = get_fold_indices(dataset, n_splits, fold, seed)[0]
        super().__init__(dataset, indices)


class ValidationSet(Subset):
    def __init__(
        self, dataset: Dataset, n_splits: int = 5, fold: int = 0, seed: int = 0
    ):
        indices = get_fold_indices(dataset, n_splits, fold, seed)[1]
        super().__init__(dataset, indices)
