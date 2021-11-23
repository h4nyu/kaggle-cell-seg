import torch
import torch.nn as nn
from torch import Tensor
from .backbones import FPNLike
from .heads import Head
from .solo import MasksToCenters, CentersToGridIndex
from .loss import FocalLoss
import torch.nn.functional as F
from typing import Callable, Any
from torchvision.ops import roi_pool, box_convert, roi_align, masks_to_boxes
from torch.cuda.amp import GradScaler, autocast
from .util import draw_save
from .assign import IoUAssign

Batch = tuple[Tensor, list[Tensor], list[Tensor]]  # id, images, mask_batch, label_batch


class Anchor:
    def __init__(
        self,
        num_classes: int,
        grid_size: int,
        box_size: int,
    ) -> None:
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.to_index = CentersToGridIndex(self.grid_size)
        self.box_size = box_size

    @torch.no_grad()
    def __call__(
        self,
        masks: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:  # category_grid, boxes, labels
        device = masks.device
        cagetory_grid = torch.zeros(
            self.num_classes, self.grid_size, self.grid_size, dtype=torch.float
        ).to(device)
        boxes = masks_to_boxes(masks)
        centers = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")[:, :2].long()
        centers, indices = torch.unique(centers, dim=0, return_inverse=True)
        box_wh = torch.ones(centers.shape).to(device) * self.box_size
        indices = torch.unique(indices, dim=0)
        labels = labels[indices]
        boxes = box_convert(
            torch.cat([centers, box_wh], dim=1), in_fmt="cxcywh", out_fmt="xyxy"
        )
        masks = masks[indices]
        mask_index = self.to_index(centers)
        index = labels * self.grid_size ** 2 + mask_index
        flattend = cagetory_grid.view(-1)
        flattend[index.long()] = 1
        cagetory_grid = flattend.view(self.num_classes, self.grid_size, self.grid_size)
        return cagetory_grid, masks, boxes, labels


class BatchAdaptor:
    def __init__(
        self,
        num_classes: int,
        grid_size: int,
        original_size: int,
        box_size: int,
    ) -> None:
        self.anchor = Anchor(
            grid_size=grid_size,
            num_classes=num_classes,
            box_size=box_size,
        )
        self.masks_to_centers = MasksToCenters()

    @torch.no_grad()
    def __call__(
        self,
        mask_batch: list[Tensor],
        label_batch: list[Tensor],
    ) -> tuple[
        Tensor, list[Tensor], list[Tensor], list[Tensor]
    ]:  # category_grids, list of mask_index, list of labels
        filtered_label_batch: list[Tensor] = []
        filtered_box_batch: list[Tensor] = []
        filtered_mask_batch: list[Tensor] = []
        cate_grids: list[Tensor] = []
        for masks, labels in zip(mask_batch, label_batch):
            (
                category_grid,
                filtered_masks,
                filtered_boxes,
                filtered_labels,
            ) = self.anchor(
                masks=masks,
                labels=labels,
            )
            filtered_label_batch.append(filtered_labels)
            filtered_box_batch.append(filtered_boxes)
            filtered_mask_batch.append(filtered_masks)
            cate_grids.append(category_grid)
        category_grids = torch.stack(cate_grids)
        return (
            category_grids,
            filtered_mask_batch,
            filtered_box_batch,
            filtered_label_batch,
        )


class GridsToBoxes:
    def __init__(
        self,
        box_size: int,
        kernel_size: int = 5,
        threshold: float = 0.95,
    ) -> None:
        self.threshold = threshold
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
        self.box_size = box_size

    def __call__(self, category_grids: Tensor) -> list[Tensor]:
        batch_size = category_grids.shape[0]
        device = category_grids.device
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
        all_box_wh = torch.ones(all_centers.size()).to(device) * self.box_size
        all_boxes = box_convert(
            torch.cat([all_centers, all_box_wh], dim=1), in_fmt="cxcywh", out_fmt="xyxy"
        )
        box_batch: list[Tensor] = []
        for batch_idx in range(batch_size):
            boxes = all_boxes[batch_indecies == batch_idx]
            box_batch.append(boxes)
        return box_batch


class CenterCrop:
    def __init__(
        self,
        box_size: int,
    ) -> None:
        self.box_size = box_size

    def __call__(self, box_batch: list[Tensor], feature: Tensor) -> Tensor:
        device = feature.device
        _, _, h, w = feature.shape
        all_patches = roi_pool(feature, box_batch, output_size=self.box_size)
        return all_patches


class CropMasks:
    def __init__(
        self,
        box_size: int,
    ) -> None:
        self.box_size = box_size
        self.pad = nn.ZeroPad2d(self.box_size // 2)

    def __call__(self, masks: Tensor, boxes: Tensor) -> Tensor:
        device = masks.device
        masks = self.pad(masks)
        boxes = boxes + self.box_size // 2
        out_masks = torch.zeros(
            (len(masks), self.box_size, self.box_size), dtype=torch.bool
        ).to(device)
        for i, b in enumerate(boxes.long()):
            out_masks[i] = masks[i, b[1] : b[3], b[0] : b[2]]
        return out_masks


class CenterSegment(nn.Module):
    def __init__(
        self,
        backbone: FPNLike,
        hidden_channels: int,
        num_classes: int,
        box_size: int,
        category_feat_range: tuple[int, int],
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
            num_classes=1,
            channels=[3],
            reductions=[1],
            use_cord=False,
        )
        self.center_crop = CenterCrop(box_size=box_size)
        self.grids_to_boxes = GridsToBoxes(box_size=box_size)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        features = self.backbone(images)
        category_feats = features[
            self.category_feat_range[0] : self.category_feat_range[1]
        ]
        category_grids = self.category_head(category_feats)
        box_batch = self.grids_to_boxes(category_grids)
        roi_feature = self.center_crop(box_batch, images)
        all_masks = self.segmentaition_head([roi_feature])[:, 0, :, :]  # feature list?
        return category_grids, all_masks, box_batch


class Criterion:
    def __init__(
        self,
        box_size: int,
        iou_threshold: float = 0.7,
        mask_weight: float = 1.0,
        category_weight: float = 1.0,
    ) -> None:
        self.category_loss = FocalLoss()
        self.mask_loss = FocalLoss()
        self.category_weight = category_weight
        self.mask_weight = mask_weight
        self.crop_masks = CropMasks(box_size)
        self.assign = IoUAssign(iou_threshold)

    def __call__(
        self,
        inputs: tuple[Tensor, Tensor, list[Tensor]],
        targets: tuple[
            Tensor, list[Tensor], list[Tensor]
        ],  # gt_cate_grids, mask_batch, mask_index_batch, filter_index_batch
    ) -> tuple[Tensor, Tensor, Tensor]:
        pred_category_grids, pred_all_masks, pred_box_batch = inputs
        gt_category_grids, gt_mask_batch, gt_box_batch = targets

        batch_size = pred_category_grids.size(0)
        device = pred_category_grids.device

        category_loss = self.category_loss(pred_category_grids, gt_category_grids)
        mask_loss = torch.tensor(0.0).to(device)

        draw_save(
            f"/store/category_pred.png",
            pred_category_grids[0],
        )
        draw_save(
            f"/store/category_gt.png",
            gt_category_grids[0],
        )

        batch_start = 0
        for i, (gt_masks, pred_boxes, gt_boxes) in enumerate(
            zip(
                gt_mask_batch,
                pred_box_batch,
                gt_box_batch,
            )
        ):
            matched = self.assign(gt_boxes, pred_boxes)
            if len(matched) > 0:
                gt_matched_masks = self.crop_masks(
                    gt_masks[matched[:, 0]], pred_boxes[matched[:, 1]]
                )
                pred_matched_masks = pred_all_masks[
                    batch_start : batch_start + len(pred_boxes)
                ][matched[:, 1]]
                empty_filter = gt_matched_masks.sum(dim=[1, 2]) > 0
                gt_matched_masks = gt_matched_masks[empty_filter]
                pred_matched_masks = pred_matched_masks[empty_filter]
                print(pred_matched_masks.shape, gt_matched_masks.shape)
                mask_loss += self.mask_loss(pred_matched_masks, gt_matched_masks)
            batch_start += len(pred_boxes)
        loss = self.category_weight * category_loss + self.mask_weight * mask_loss
        return loss, category_loss, mask_loss


class TrainStep:
    def __init__(
        self,
        criterion: Criterion,
        model: CenterSegment,
        optimizer: Any,
        batch_adaptor: BatchAdaptor,
        use_amp: bool = True,
    ) -> None:
        self.criterion = criterion
        self.model = model
        self.bath_adaptor = batch_adaptor
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.scaler = GradScaler()

    def __call__(self, batch: Batch) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            images, gt_mask_batch, gt_label_batch = batch
            gt_category_grids, gt_mask_batch, gt_box_batch, _ = self.bath_adaptor(
                mask_batch=gt_mask_batch, label_batch=gt_label_batch
            )
            pred_category_grids, pred_all_masks, pred_box_batch = self.model(images)
            loss, category_loss, mask_loss = self.criterion(
                (
                    pred_category_grids,
                    pred_all_masks,
                    pred_box_batch,
                ),
                (
                    gt_category_grids,
                    gt_mask_batch,
                    gt_box_batch,
                ),
            )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return dict(
            loss=loss.item(),
            category_loss=category_loss.item(),
            mask_loss=mask_loss.item(),
        )
