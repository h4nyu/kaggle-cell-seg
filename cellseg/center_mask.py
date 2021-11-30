import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from .heads import Head
from typing import Protocol, TypedDict, Optional, Callable, Any
from cellseg.loss import FocalLoss, DiceLoss
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import masks_to_boxes, box_convert, roi_align
from .utils import grid, draw_save, ToPatches, MergePatchedMasks
from .backbones import FPNLike

Batch = tuple[Tensor, list[Tensor], list[Tensor]]  # id, images, mask_batch, label_batch


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
            use_cord=False,
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
            1, self.patch_size, self.patch_size, dtype=torch.float
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
        sliency_mask = masks.sum(dim=0).view(1, self.patch_size, self.patch_size).bool()
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
        self.size_loss = nn.SmoothL1Loss()
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
        size_loss = self.size_loss(
            pred_size_grids.masked_select(pos_masks),
            gt_size_grids.masked_select(pos_masks),
        )
        offset_loss = self.offset_loss(
            pred_offset_grids.masked_select(pos_masks),
            gt_offset_grids.masked_select(pos_masks),
        )
        mask_loss = self.mask_loss(
            pred_mask_grids.masked_select(pos_masks),
            gt_mask_grids.masked_select(pos_masks),
        )
        sliency_loss = self.sliency_loss(pred_size_grids, gt_size_grids)
        loss = (
            self.category_weight * category_loss
            + self.size_weight * size_loss
            + self.offset_weight * offset_loss
            + self.mask_weight * mask_loss
            + self.sliency_weight * sliency_loss
        )
        return loss, category_loss, size_loss, offset_loss, mask_loss, sliency_loss


class ToMasks:
    def __init__(
        self,
        category_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        kernel_size: int = 3,
    ) -> None:
        self.category_threshold = category_threshold
        self.mask_threshold = mask_threshold
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )

    def __call__(
        self,
        category_grids: Tensor,
        size_grids: Tensor,
        offset_grids: Tensor,
        mask_grids: Tensor,
        sliency_masks: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        device = category_grids.device
        batch_size, _, grid_size = category_grids.shape[:3]
        patch_size = sliency_masks.shape[2]
        mask_size = int(math.sqrt(mask_grids.shape[1]))
        reduction = patch_size // grid_size
        category_grids = category_grids * (
            (category_grids > self.category_threshold)
            & (self.max_pool(category_grids) == category_grids)
        )
        (
            batch_indecies,
            all_labels,
            all_cy,
            all_cx,
        ) = category_grids.nonzero(as_tuple=True)
        all_cxcy = torch.stack([all_cx, all_cy], dim=1)
        mask_batch: list[Tensor] = []
        label_batch: list[Tensor] = []
        score_batch: list[Tensor] = []
        for batch_idx in range(batch_size):
            batch_filter = batch_indecies == batch_idx
            labels = all_labels[batch_filter]
            cxcy_index = all_cxcy[batch_filter]
            scores = category_grids[
                batch_idx, labels, cxcy_index[:, 1], cxcy_index[:, 0]
            ]
            box_sizes = size_grids[batch_idx, :, cxcy_index[:, 1], cxcy_index[:, 0]].t()
            cxcy_offsets = offset_grids[
                batch_idx, :, cxcy_index[:, 1], cxcy_index[:, 0]
            ].t()
            cxcywhs = torch.cat(
                [(cxcy_index + cxcy_offsets) * reduction, box_sizes * patch_size], dim=1
            )
            boxes = (
                box_convert(cxcywhs, in_fmt="cxcywh", out_fmt="xyxy")
                .round()
                .long()
                .clip(min=0, max=patch_size - 1)
            )
            grid_masks = mask_grids[
                batch_idx, :, cxcy_index[:, 1], cxcy_index[:, 0]
            ].t()
            sliency_mask = sliency_masks[batch_idx]
            masks = torch.zeros(len(boxes), patch_size, patch_size).to(device)
            for i, (b, gm) in enumerate(zip(boxes, grid_masks)):
                crop_mask = torch.zeros(sliency_mask.shape).to(device)
                gm = F.interpolate(
                    gm.view(1, 1, mask_size, mask_size),
                    size=(
                        b[3] - b[1] + 1,
                        b[2] - b[0] + 1,
                    ),
                )
                crop_mask[:, b[1] : b[3] + 1, b[0] : b[2] + 1] = gm.view(
                    1, *gm.shape[2:]
                )
                masks[i] = sliency_mask * crop_mask

            masks = masks > self.mask_threshold
            empty_filter = masks.sum(dim=[1, 2]) > 0
            masks = masks[empty_filter]
            scores = scores[empty_filter]
            labels = labels[empty_filter]
            label_batch.append(labels)
            mask_batch.append(masks)
            score_batch.append(scores)
        return mask_batch, label_batch, score_batch


class TrainStep:
    def __init__(
        self,
        criterion: Criterion,
        model: CenterMask,
        optimizer: Any,
        batch_adaptor: BatchAdaptor,
        use_amp: bool = True,
        to_masks: Optional[ToMasks] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.criterion = criterion
        self.model = model
        self.batch_adaptor = batch_adaptor
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.scaler = GradScaler()
        self.to_masks = to_masks
        self.scheduler = scheduler

    def __call__(self, batch: Batch) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            images, gt_mask_batch, gt_label_batch = batch
            (
                gt_category_grids,
                gt_size_grids,
                gt_offset_grids,
                gt_mask_grids,
                gt_sliency_masks,
                pos_masks,
            ) = self.batch_adaptor(mask_batch=gt_mask_batch, label_batch=gt_label_batch)
            (
                pred_category_grids,
                pred_size_grids,
                pred_offset_grids,
                pred_mask_grids,
                pred_sliency_masks,
            ) = self.model(images)
            (
                loss,
                category_loss,
                size_loss,
                offset_loss,
                mask_loss,
                sliency_loss,
            ) = self.criterion(
                (
                    pred_category_grids,
                    pred_size_grids,
                    pred_offset_grids,
                    pred_mask_grids,
                    pred_sliency_masks,
                ),
                (
                    gt_category_grids,
                    gt_size_grids,
                    gt_offset_grids,
                    gt_mask_grids,
                    gt_sliency_masks,
                    pos_masks,
                ),
            )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step(loss)

        if self.to_masks is not None:
            pred_mask_batch, _, _ = self.to_masks(
                pred_category_grids,
                pred_size_grids,
                pred_offset_grids,
                pred_mask_grids,
                pred_sliency_masks,
            )
            draw_save(
                "/app/test_outputs/gt.png",
                images[0],
                gt_mask_batch[0],
            )
            draw_save(
                "/app/test_outputs/pred.png",
                images[0],
                pred_mask_batch[0],
            )

        return dict(
            loss=loss.item(),
            category_loss=category_loss.item(),
            size_loss=size_loss.item(),
            offset_loss=offset_loss.item(),
            mask_loss=mask_loss.item(),
            sliency_loss=sliency_loss.item(),
        )


class ValidationStep:
    def __init__(
        self,
        criterion: Criterion,
        model: CenterMask,
        batch_adaptor: BatchAdaptor,
        to_masks: ToMasks,
    ) -> None:
        self.criterion = criterion
        self.model = model
        self.batch_adaptor = batch_adaptor
        self.to_masks = to_masks

    @torch.no_grad()
    def __call__(
        self,
        batch: Batch,
        on_end: Optional[Callable[[list[Tensor], list[Tensor]], Any]] = None,
    ) -> dict[str, float]:  # logs
        self.model.eval()
        images, gt_mask_batch, gt_label_batch = batch
        (
            gt_category_grids,
            gt_size_grids,
            gt_offset_grids,
            gt_mask_grids,
            gt_sliency_masks,
            pos_masks,
        ) = self.batch_adaptor(mask_batch=gt_mask_batch, label_batch=gt_label_batch)
        (
            pred_category_grids,
            pred_size_grids,
            pred_offset_grids,
            pred_mask_grids,
            pred_sliency_masks,
        ) = self.model(images)
        (
            loss,
            category_loss,
            size_loss,
            offset_loss,
            mask_loss,
            sliency_loss,
        ) = self.criterion(
            (
                pred_category_grids,
                pred_size_grids,
                pred_offset_grids,
                pred_mask_grids,
                pred_sliency_masks,
            ),
            (
                gt_category_grids,
                gt_size_grids,
                gt_offset_grids,
                gt_mask_grids,
                gt_sliency_masks,
                pos_masks,
            ),
        )
        if on_end is not None:
            pred_mask_batch, pred_label_batch, _ = self.to_masks(
                pred_category_grids,
                pred_size_grids,
                pred_offset_grids,
                pred_mask_grids,
                pred_sliency_masks,
            )
            on_end(pred_mask_batch, gt_mask_batch)
        return dict(
            loss=loss.item(),
            category_loss=category_loss.item(),
            size_loss=size_loss.item(),
            offset_loss=offset_loss.item(),
            mask_loss=mask_loss.item(),
            sliency_loss=sliency_loss.item(),
        )
