from torchvision.utils import draw_segmentation_masks, save_image, draw_bounding_boxes
from torchvision.ops import masks_to_boxes
import numpy as np
import torch
from torch import Tensor
import random
import torch.nn as nn
from typing import Optional, TypeVar, Generic, Any, Union
from omegaconf import OmegaConf
from pathlib import Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


@torch.no_grad()
def grid(h: int, w: int, dtype: Optional[torch.dtype] = None) -> tuple[Tensor, Tensor]:
    grid_y, grid_x = torch.meshgrid(  # type:ignore
        torch.arange(h, dtype=dtype),
        torch.arange(w, dtype=dtype),
    )
    return (grid_y, grid_x)


T = TypeVar("T", bound=nn.Module)


class Checkpoint(Generic[T]):
    def __init__(self, root_path: str, default_score: float) -> None:
        self.root_path = Path(root_path)
        self.model_path = self.root_path.joinpath("checkpoint.pth")
        self.checkpoint_path = self.root_path.joinpath("checkpoint.yaml")
        self.default_score = default_score
        self.root_path.mkdir(exist_ok=True)

    def load_if_exists(self, model: T) -> tuple[T, float]:
        if self.model_path.exists() and self.checkpoint_path.exists():
            model.load_state_dict(torch.load(self.model_path))
            conf = OmegaConf.load(self.checkpoint_path)
            score = conf.get("score", self.default_score)  # type: ignore
            return model, score
        else:
            return model, self.default_score

    def save(self, model: T, score: float) -> float:
        torch.save(model.state_dict(), self.model_path)  # type: ignore
        OmegaConf.save(config=dict(score=score), f=self.checkpoint_path)
        return score


class MeanReduceDict:
    def __init__(self, keys: list[str] = []) -> None:
        self.keys = keys
        self.running: dict[str, float] = {}
        self.num_samples = 0

    def accumulate(self, log: dict[str, float]) -> None:
        for k in self.keys:
            self.running[k] = self.running.get(k, 0) + log[k]
        self.num_samples += 1

    @property
    def value(self) -> dict[str, float]:
        return {k: v / max(1, self.num_samples) for k, v in self.running.items()}


@torch.no_grad()
def draw_save(
    path: str,
    image: Tensor,
    masks: Optional[Tensor] = None,
) -> None:
    image = image.detach().to("cpu").float()
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if masks is not None:
        plot = draw_segmentation_masks((image * 255).to(torch.uint8), masks, alpha=0.3)
        empty_filter = masks.sum(dim=[1, 2]) > 0
        masks = masks[empty_filter]
        masks = masks.to("cpu")
        if len(masks) > 0:
            boxes = masks_to_boxes(masks)
            plot = draw_bounding_boxes(plot, boxes)
        plot = plot / 255
    else:
        plot = image
    save_image(plot, path)
