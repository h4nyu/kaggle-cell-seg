import torch
from torch import Tensor
from typing import Optional
from torchvision.ops import masks_to_boxes, box_convert


def grid(h: int, w: int, dtype: Optional[torch.dtype] = None) -> tuple[Tensor, Tensor]:
    grid_y, grid_x = torch.meshgrid(  # type:ignore
        torch.arange(h, dtype=dtype),
        torch.arange(w, dtype=dtype),
    )
    return (grid_y, grid_x)


class MasksToCenters:
    def __call__(
        self,
        masks: Tensor,
    ) -> Tensor:
        boxes = masks_to_boxes(masks)
        return box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")[:, :2]


class CentersToGridIndex:
    def __init__(
        self,
        grid_size: int,
    ) -> None:
        self.grid_size = grid_size

    def __call__(
        self,
        centers: Tensor,
    ) -> Tensor:
        return (centers[:, 1] * self.grid_size + centers[:, 0]).long()


class ToCategoryGrid:
    def __init__(
        self,
        num_classes: int,
        grid_size: int,
    ) -> None:
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.to_index = CentersToGridIndex(self.grid_size)

    def __call__(
        self,
        centers: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:  # category_grid, mask_index
        device = centers.device
        cagetory_grid = torch.zeros(
            self.num_classes, self.grid_size, self.grid_size, dtype=torch.float32
        ).to(device)
        mask_index = self.to_index(centers)
        index = labels * self.grid_size ** 2 + mask_index
        flattend = cagetory_grid.view(-1)
        flattend[index.long()] = 1
        cagetory_grid = flattend.view(self.num_classes, self.grid_size, self.grid_size)
        return cagetory_grid, mask_index


class BatchAdaptor:
    def __init__(
        self,
        num_classes: int,
        grid_size: int,
        original_size: int,
    ) -> None:
        self.grid_size = grid_size
        self.to_category_grid = ToCategoryGrid(
            grid_size=grid_size,
            num_classes=num_classes,
        )
        self.scale = grid_size / original_size
        self.masks_to_centers = MasksToCenters()

    def __call__(
        self,
        mask_batch: list[Tensor],
        label_batch: list[Tensor],
    ) -> tuple[Tensor, list[Tensor]]:  # category_grids, list of mask_index
        batch: list[Tensor] = []
        cate_grids: list[Tensor] = []
        for masks, labels in zip(mask_batch, label_batch):
            scaled_centers = self.masks_to_centers(masks) * self.scale
            category_grid, mask_index = self.to_category_grid(
                centers=scaled_centers,
                labels=labels,
            )
            cate_grids.append(category_grid)
            batch.append(mask_index)
        category_grids = torch.stack(cate_grids)
        return category_grids, batch
