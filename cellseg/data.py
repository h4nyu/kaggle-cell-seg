from typing import Optional
import numpy as np
import torch
from torch import Tensor
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks, save_image
import pandas as pd
from typing import TypedDict, Optional, cast, Callable, Any, Union
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
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


def draw_save(
    path: str,
    image: Tensor,
    masks: Optional[Tensor] = None,
) -> None:
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if masks is not None:
        plot = (
            draw_segmentation_masks((image * 255).to(torch.uint8), masks, alpha=0.3)
            / 255
        )
    else:
        plot = image
    save_image(plot, path)


class ToDevice:
    def __init__(
        self,
        device: str,
    ) -> None:
        self.device = device

    def __call__(self, *args: Union[Tensor, list[Tensor]]) -> Any:
        return tuple(
            [i.to(self.device) for i in x] if isinstance(x, list) else x.to(self.device)
            for x in args
        )


TrainItem = TypedDict(
    "TrainItem",
    {
        "id": str,
        "image": Tensor,
        "masks": Tensor,
        "labels": Tensor,
    },
)


class Tranform:
    def __init__(self, original_size: int):

        self.transform = A.Compose(
            [
                A.Resize(
                    width=original_size,
                    height=original_size,
                    interpolation=cv2.INTER_LINEAR,
                    p=1,
                ),
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


def get_fold_indices(
    dataset: Dataset, n_splits: int = 5, index: int = 0, seed: int = 0
) -> tuple[list[int], list[int]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x = np.arange(len(dataset))
    y = dataset.stratums
    return list(splitter.split(x, y))[index]


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
        labels = torch.from_numpy(transformed["labels"])
        return dict(
            id=image_id,
            image=image,
            masks=masks,
            labels=labels,
        )
