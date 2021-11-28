import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .heads import Head
from typing import Protocol, TypedDict, Optional, Callable, Any
from cellseg.loss import FocalLoss, DiceLoss
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import masks_to_boxes, box_convert, roi_align
from .utils import grid, draw_save, ToPatches, MergePatchedMasks
from .backbones import FPNLike


class CenterMask(nn.Module):
    def __init__(
        self,
        backbone: FPNLike,
        hidden_channels: int,
        mask_size: int,
        category_feat_range: tuple[int, int],
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.category_feat_range = category_feat_range
        self.backbone = backbone
        self.category_head = Head(
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=False,
        )

        self.size_head = Head(
            hidden_channels=hidden_channels,
            num_classes=2,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=False,
        )

        self.offset_head = Head(
            hidden_channels=hidden_channels,
            num_classes=2,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=False,
        )

        self.sliency_head = Head(
            hidden_channels=hidden_channels,
            num_classes=1,
            channels=backbone.channels,
            reductions=backbone.reductions,
            use_cord=False,
        )

        self.mask_head = Head(
            hidden_channels=hidden_channels,
            num_classes=mask_size ** 2,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
            use_cord=True,
        )

    def forward(
        self, image_batch: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        features = self.backbone(image_batch)
        category_feats = features[
            self.category_feat_range[0] : self.category_feat_range[1]
        ]
        category_grids = self.category_head(category_feats)
        size_grids = self.size_head(category_feats)
        offset_grids = self.offset_head(category_feats)
        mask_grids = self.mask_head(category_feats)
        sliency_masks = self.sliency_head(features)
        return (category_grids, size_grids, offset_grids, mask_grids, sliency_masks)


class BatchAdaptor:
    def __init__(
        self,
        num_classes: int,
        grid_size: int,
        mask_size: int,
        patch_size: int,
    ) -> None:
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.mask_size = mask_size
        self.patch_size = patch_size
        self.reduction = patch_size // grid_size
        self.mask_area = mask_size ** 2

    def mkgrid(
        self,
        masks: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        device = masks.device
        category_grid = torch.zeros(
            self.num_classes, self.grid_size, self.grid_size, dtype=torch.float
        ).to(device)
        size_grid = torch.zeros(
            2, self.grid_size, self.grid_size, dtype=torch.float
        ).to(device)
        offset_grid = torch.zeros(
            2, self.grid_size, self.grid_size, dtype=torch.float
        ).to(device)
        mask_grid = torch.zeros(
            self.mask_size ** 2, self.grid_size, self.grid_size, dtype=torch.float
        ).to(device)
        sliency_mask = torch.zeros(
            self.mask_size ** 2, self.grid_size, self.patch_size, dtype=torch.float
        ).to(device)
        pos_mask = torch.zeros(1, self.grid_size, self.grid_size, dtype=torch.bool).to(
            device
        )
        if len(masks) == 0:
            return (
                category_grid,
                size_grid,
                offset_grid,
                mask_grid,
                sliency_mask,
                pos_mask,
            )

        boxes = masks_to_boxes(masks)
        cxcy_boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
        for mask, label, box, cxcywh in zip(masks, labels, boxes.long(), cxcy_boxes):
            cxcy_index = (cxcywh[:2] / self.reduction).long()
            category_grid[label, cxcy_index[1], cxcy_index[0]] = 1
            size_grid[:, cxcy_index[1], cxcy_index[0]] = cxcywh[2:]
            offset_grid[:, cxcy_index[1], cxcy_index[0]] = (
                cxcywh[:2] / self.reduction - cxcy_index
            )
            mask_grid[:, cxcy_index[1], cxcy_index[0]] = F.interpolate(
                mask[box[1] : box[3], box[0] : box[2]]
                .view(1, 1, cxcywh[3].long(), cxcywh[2].long())
                .float(),
                size=(self.mask_size, self.mask_size),
            ).view(self.mask_area)
        sliency_mask = masks.sum(dim=0).view(1, self.patch_size, self.patch_size)
        pos_mask = (
            category_grid.sum(dim=0).view(1, self.grid_size, self.grid_size).bool()
        )
        size_grid = size_grid / self.patch_size
        return category_grid, size_grid, offset_grid, mask_grid, sliency_mask, pos_mask

    @torch.no_grad()
    def __call__(
        self,
        mask_batch: list[Tensor],
        label_batch: list[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        cate_grids: list[Tensor] = []
        size_grids: list[Tensor] = []
        offset_grids: list[Tensor] = []
        mask_grids: list[Tensor] = []
        sliency_masks: list[Tensor] = []
        pos_masks: list[Tensor] = []
        for masks, labels in zip(mask_batch, label_batch):
            cate, size, offset, mask, sliency, pos = self.mkgrid(
                masks=masks,
                labels=labels,
            )
            cate_grids.append(cate)
            size_grids.append(size)
            offset_grids.append(offset)
            mask_grids.append(mask)
            sliency_masks.append(sliency)
            pos_masks.append(pos)
        return (
            torch.stack(cate_grids),
            torch.stack(size_grids),
            torch.stack(offset_grids),
            torch.stack(mask_grids),
            torch.stack(sliency_masks),
            torch.stack(pos_masks),
        )


class Criterion:
    def __init__(
        self,
        mask_weight: float = 1.0,
        sliency_weight: float = 1.0,
        category_weight: float = 1.0,
        size_weight: float = 1.0,
        offset_weight: float = 1.0,
    ) -> None:
        self.category_loss = FocalLoss()
        self.size_loss = nn.MSELoss()
        self.offset_loss = nn.MSELoss()
        self.mask_loss = FocalLoss()
        self.sliency_loss = FocalLoss()
        self.category_weight = category_weight
        self.mask_weight = mask_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight
        self.sliency_weight = sliency_weight

    def __call__(
        self,
        inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        targets: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            pred_category_grids,
            pred_size_grids,
            pred_offset_grids,
            pred_mask_grids,
            pred_sliency_masks,
        ) = inputs
        (
            gt_category_grids,
            gt_size_grids,
            gt_offset_grids,
            gt_mask_grids,
            gt_sliency_masks,
            pos_masks,
        ) = targets
        device = pred_category_grids.device
        category_loss = self.category_loss(pred_category_grids, gt_category_grids)
        size_loss = self.size_loss(pred_size_grids, gt_size_grids)
        offset_loss = self.offset_loss(pred_offset_grids, gt_offset_grids)
        mask_loss = self.mask_loss(pred_mask_grids, gt_mask_grids)
        sliency_loss = self.sliency_loss(pred_size_grids, gt_size_grids)
        loss = (
            self.category_weight * category_loss
            + self.size_weight * size_loss
            + self.offset_weight * offset_loss
            + self.mask_weight * mask_loss
            + self.sliency_weight * sliency_loss
        )
        return loss, category_loss, size_loss, offset_loss, mask_loss, sliency_loss
