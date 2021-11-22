import torch
import torch.nn as nn
from torch import Tensor
from .backbones import FPNLike
from .heads import Head
from typing import Callable
from torchvision.ops import roi_pool, box_convert, roi_align


class GridsToCenters:
    def __init__(self, kernel_size: int = 3, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

    def __call__(self, category_grids: Tensor) -> list[Tensor]:
        batch_size = category_grids.shape[0]
        category_grids = category_grids * (
            (self.max_pool(category_grids) == category_grids)
            & (category_grids > self.threshold)
        )
        (
            batch_indecies,
            labels,
            cy,
            cx,
        ) = category_grids.nonzero().unbind(-1)
        all_centers = torch.stack([cx, cy], dim=1)
        center_batch: list[Tensor] = []
        for batch_idx in range(batch_size):
            centers = all_centers[batch_indecies == batch_idx]
            center_batch.append(centers)
        return center_batch

        #     masks = all_masks[batch_idx][mask_indecies[filterd]]
        #     empty_filter = masks.sum(dim=[1, 2]) > 0


class CenterCrop:
    def __init__(
        self,
        output_size: int,
    ) -> None:
        self.output_size = output_size

    def __call__(self, center_batch: list[Tensor], images: Tensor) -> Tensor:
        device = images.device
        _, _, h, w = images.shape
        box_batch = []
        for centers in center_batch:
            box_wh = torch.ones(centers.size()).to(device) * self.output_size
            boxes = box_convert(
                torch.cat([centers, box_wh], dim=1), in_fmt="cxcywh", out_fmt="xyxy"
            )
            box_batch.append(boxes)
        return roi_pool(images, box_batch, output_size=self.output_size)


class TrainCenterCrop:
    def __init__(
        self,
        out_size: int,
    ) -> None:
        ...
        self.out_size = out_size

    def __call__(self, center_batch: list[Tensor], images: Tensor) -> Tensor:

        return []


class CenterSegment(nn.Module):
    def __init__(
        self,
        backbone: FPNLike,
        hidden_channels: int,
        num_classes: int,
        category_feat_range: tuple[int, int],
        center_crop: Callable[[list[Tensor], Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.category_feat_range = category_feat_range

        self.category_head = Head(
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=False,
        )
        self.segmentaition_head = Head(
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            channels=[3],
            reductions=[1],
            use_cord=False,
        )
        self.center_crop = center_crop
        self.grids_to_centers = GridsToCenters()
        self.a = nn.Conv2d(3, 3, kernel_size=3)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        features = self.backbone(images)
        category_feats = features[
            self.category_feat_range[0] : self.category_feat_range[1]
        ]
        category_grids = self.category_head(category_feats)
        center_batch = self.grids_to_centers(category_grids)
        croped_images = self.center_crop(center_batch, images)
        masks = self.segmentaition_head([croped_images])  # feature list?
        return category_grids, masks, center_batch
