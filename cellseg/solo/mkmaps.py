import torch
from typing import Literal, Optional
from torch import Tensor

GaussianMapMode = Literal["length", "aspect", "constant"]


class MkMapsBase:
    num_classes: int

    def _mkmaps(
        self,
        boxes: Tensor,
        hw: tuple[int, int],
        original_hw: tuple[int, int],
    ) -> Tensor:
        ...

    @torch.no_grad()
    def __call__(
        self,
        box_batch: list[Tensor],
        label_batch: list[Tensor],
        hw: tuple[int, int],
        original_hw: tuple[int, int],
    ) -> Tensor:
        hms: list[torch.Tensor] = []
        for boxes, labels in zip(box_batch, label_batch):
            hm = torch.cat(
                [
                    self._mkmaps(Tensor(boxes[labels == i]), hw, original_hw)
                    for i in range(self.num_classes)
                ],
                dim=1,
            )
            hms.append(hm)
        return torch.cat(hms, dim=0)


class MkGaussianMaps(MkMapsBase):
    def __init__(
        self,
        num_classes: int,
        sigma: float = 0.1,
        mode: Optional[GaussianMapMode] = None,
    ) -> None:
        self.sigma = sigma
        self.mode = mode
        self.num_classes = num_classes

    def _mkmaps(
        self,
        boxes: Tensor,
        hw: tuple[int, int],
        original_hw: tuple[int, int],
    ) -> Tensor:
        device = boxes.device
        h, w = hw
        orig_h, orig_w = original_hw
        heatmap = torch.zeros((1, 1, h, w), dtype=torch.float32).to(device)
        box_count = len(boxes)
        if box_count == 0:
            return heatmap
        grid_y, grid_x = torch.meshgrid(  # type:ignore
            torch.arange(h, dtype=torch.int64),
            torch.arange(w, dtype=torch.int64),
        )
        img_wh = torch.tensor([w, h]).to(device)
        x0, y0, x1, y1 = boxes.unbind(-1)
        cxcy = torch.stack(
            [
                (x1 + x0) / 2 * w / orig_w,
                (y1 + y0) / 2 * h / orig_h,
            ],
            dim=1,
        )
        box_wh = torch.stack(
            [
                (x1 - x0) / orig_w,
                (y1 - y0) / orig_h,
            ],
            dim=1,
        )
        grid_xy = torch.stack([grid_x, grid_y]).to(device).expand((box_count, 2, h, w))
        grid_cxcy = cxcy.view(box_count, 2, 1, 1).expand_as(grid_xy)
        if self.mode == "aspect":
            weight = (box_wh ** 2).clamp(min=1e-4).view(box_count, 2, 1, 1)
        elif self.mode == "length":
            weight = (
                (boxes[:, 2:] ** 2)
                .min(dim=1, keepdim=True)[0]
                .clamp(min=1e-4)
                .view(box_count, 1, 1, 1)
            )
        else:
            weight = torch.ones((box_count, 1, 1, 1)).to(device)
        mounts = torch.exp(
            -(((grid_xy - grid_cxcy.long()) ** 2) / weight).sum(dim=1, keepdim=True)
            / (2 * self.sigma ** 2)
        )
        heatmap, _ = mounts.max(dim=0, keepdim=True)
        return heatmap
