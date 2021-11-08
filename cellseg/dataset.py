import torch
from torch.utils.data import Dataset
from torch import Tensor
from cellseg.data import get_masks
import pandas as pd
from torchvision.io import read_image
from typing import TypedDict, Optional, cast

TrainItem = TypedDict(
    "TrainItem",
    {
        "id": str,
        "image": Tensor,
        "masks": Tensor,
        "labels": Tensor,
    },
)


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
