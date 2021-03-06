from torchvision.utils import draw_segmentation_masks, save_image, draw_bounding_boxes
from torchvision.ops import masks_to_boxes
import numpy as np
import torch
from torch import Tensor
import random
import torch.nn as nn
from typing import Optional, TypeVar, Generic, Any, Union, Callable
from omegaconf import OmegaConf
from pathlib import Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def round_to(n: float, multiple: int = 8) -> int:
    return int(round(n / multiple) * multiple)


def weighted_mean(x: Tensor, weights: Tensor, eps: float = 1e-6) -> Tensor:
    return (x * weights).sum() / (weights.sum() + eps)


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


def grid_points(
    h: int, w: int, dtype: Optional[torch.dtype] = None
) -> Tensor:  # [h*w, 2]
    return torch.stack(
        torch.meshgrid([torch.arange(h), torch.arange(w)])[::-1], dim=2
    ).reshape(h * w, 2)


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
            self.running[k] = self.running.get(k, 0) + log.get(k, 0)
        self.num_samples += 1

    @property
    def value(self) -> dict[str, float]:
        return {k: self.running.get(k, 0) / max(1, self.num_samples) for k in self.keys}


@torch.no_grad()
def draw_save(
    path: str,
    image: Tensor,
    masks: Optional[Tensor] = None,
    boxes: Optional[Tensor] = None,
) -> None:
    image = image.detach().to("cpu").float()
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if masks is not None and len(masks) > 0:
        empty_filter = masks.sum(dim=[1, 2]) > 0
        masks = masks[empty_filter]
        masks = masks.to("cpu")
        plot = draw_segmentation_masks((image * 255).to(torch.uint8), masks, alpha=0.3)
        boxes = masks_to_boxes(masks)
        plot = draw_bounding_boxes(plot, boxes)
        plot = plot / 255
    elif boxes is not None and len(boxes) > 0:
        plot = draw_bounding_boxes((image * 255).to(torch.uint8), boxes)
        plot = plot / 255
    else:
        plot = image
    save_image(plot, path)


class ToPatches:
    def __init__(
        self,
        patch_size: int,
        use_reflect: bool = False,
    ) -> None:
        self.patch_size = patch_size
        self.use_reflect = use_reflect

    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        device = images.device
        b, c, h, w = images.shape
        pad_size = (
            0,
            (w // self.patch_size + 1) * self.patch_size - w,
            0,
            (h // self.patch_size + 1) * self.patch_size - h,
        )
        if self.use_reflect:
            pad: Callable[[Tensor], Tensor] = nn.ReflectionPad2d(pad_size)

        else:
            pad = nn.ZeroPad2d(pad_size)
        images = pad(images)
        _, _, padded_h, padded_w = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches = (
            patches.permute([0, 2, 3, 1, 4, 5])
            .contiguous()
            .view(b, -1, 3, self.patch_size, self.patch_size)
        )
        index = torch.stack(
            grid(padded_w // self.patch_size, padded_h // self.patch_size)
        ).to(device)
        patch_grid = index.permute([2, 1, 0]).contiguous().view(-1, 2) * self.patch_size
        return images, patches, patch_grid


class MergePatchedMasks:
    def __init__(
        self,
        patch_size: int,
    ) -> None:
        self.patch_size = patch_size

    def __call__(self, mask_batch: list[Tensor], patch_grid: Tensor) -> Tensor:
        device = patch_grid.device
        last_grid = patch_grid[-1]
        out_size = last_grid + self.patch_size
        out_batch: list[Tensor] = []
        for masks, grid in zip(mask_batch, patch_grid):
            out_masks = torch.zeros(
                (len(masks), int(out_size[1]), int(out_size[0])), dtype=masks.dtype
            ).to(device)
            out_masks[
                :,
                grid[1] : grid[1] + self.patch_size,
                grid[0] : grid[0] + self.patch_size,
            ] = masks
            out_batch.append(out_masks)
        return torch.cat(out_batch)
