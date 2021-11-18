import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .heads import Head
from typing import Protocol, TypedDict, Optional, Callable
from cellseg.loss import FocalLoss, DiceLoss
from .adaptors import BatchAdaptor, CentersToGridIndex
from torch.cuda.amp import GradScaler, autocast
from typing import Any


class FPNLike(Protocol):
    channels: list[int]
    reductions: list[int]

    def __call__(self, x: Tensor) -> list[Tensor]:
        ...


Batch = tuple[Tensor, list[Tensor], list[Tensor]]  # id, images, mask_batch, label_batch


class Criterion:
    def __init__(
        self,
        mask_weight: float = 1.0,
        category_weight: float = 1.0,
    ) -> None:
        self.category_loss = FocalLoss()
        self.mask_loss = DiceLoss()
        self.category_weight = category_weight
        self.mask_weight = mask_weight

    def __call__(
        self,
        inputs: tuple[Tensor, Tensor],  # pred_cate_grids, all_masks
        targets: tuple[
            Tensor, list[Tensor], list[Tensor]
        ],  # gt_cate_grids, mask_batch, mask_index_batch
    ) -> tuple[Tensor, Tensor, Tensor]:
        pred_category_grids, all_masks = inputs
        gt_category_grids, gt_mask_batch, mask_index_batch = targets
        device = pred_category_grids.device
        category_loss = self.category_loss(
            inputs=pred_category_grids, targets=gt_category_grids
        )
        mask_loss = torch.tensor(0.0).to(device)
        for gt_masks, mask_index, pred_masks in zip(
            gt_mask_batch, mask_index_batch, all_masks
        ):
            filtered_masks = pred_masks[mask_index]
            mask_loss += self.mask_loss(inputs=filtered_masks, targets=gt_masks)
        loss = self.category_weight * category_loss + self.mask_weight * mask_loss
        loss = mask_loss
        return loss, category_loss, mask_loss


class Solo(nn.Module):
    def __init__(
        self,
        backbone: FPNLike,
        hidden_channels: int,
        grid_size: int,
        category_feat_range: tuple[int, int],
        mask_feat_range: tuple[int, int],
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.category_feat_range = category_feat_range
        self.mask_feat_range = mask_feat_range
        self.backbone = backbone
        self.grid_size = grid_size
        self.category_head = Head(
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            channels=backbone.channels[category_feat_range[0] : category_feat_range[1]],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
        )

        self.mask_head = Head(
            hidden_channels=hidden_channels,
            num_classes=grid_size ** 2,
            channels=backbone.channels[mask_feat_range[0] : mask_feat_range[1]],
            reductions=backbone.reductions[mask_feat_range[0] : mask_feat_range[1]],
        )

    def forward(self, image_batch: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(image_batch)
        category_feats = features[
            self.category_feat_range[0] : self.category_feat_range[1]
        ]
        category_grid = self.category_head(category_feats)
        if category_grid.shape[2:] != (self.grid_size, self.grid_size):
            category_grid = F.interpolate(
                category_grid, size=(self.grid_size, self.grid_size)
            )
        mask_feats = features[self.mask_feat_range[0] : self.mask_feat_range[1]]
        masks = self.mask_head(mask_feats)
        return (category_grid, masks)


class ToMasks:
    def __init__(
        self,
        category_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> None:
        self.category_threshold = category_threshold
        self.mask_threshold = mask_threshold

    @torch.no_grad()
    def __call__(
        self, category_grids: Tensor, all_masks: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:  # mask_batch, label_batch, mask_indecies
        batch_size = category_grids.shape[0]
        grid_size = category_grids.shape[2]
        to_index = CentersToGridIndex(grid_size=grid_size)
        all_masks = all_masks > self.mask_threshold
        batch_indecies, labels, cy, cx, = (
            (category_grids > self.category_threshold).nonzero().unbind(-1)
        )
        mask_indecies = to_index(torch.stack([cx, cy], dim=1))
        mask_batch: list[Tensor] = []
        label_batch: list[Tensor] = []
        for batch_idx in range(batch_size):
            filterd = batch_indecies == batch_idx
            label_batch.append(labels[filterd])
            mask_batch.append(all_masks[batch_idx][mask_indecies[filterd]])

        return mask_batch, label_batch


class TrainStep:
    def __init__(
        self,
        criterion: Criterion,
        model: Solo,
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
        with autocast(enabled=self.use_amp):
            images, gt_mask_batch, gt_label_batch = batch
            gt_category_grids, mask_index = self.bath_adaptor(
                mask_batch=gt_mask_batch, label_batch=gt_label_batch
            )
            pred_category_grids, pred_all_masks = self.model(images)
            loss, category_loss, mask_loss = self.criterion(
                (
                    pred_category_grids,
                    pred_all_masks,
                ),
                (
                    gt_category_grids,
                    gt_mask_batch,
                    mask_index,
                ),
            )
            self.scaler.scale(loss).backward()
            self.optimizer.zero_grad()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        return dict(
            loss=loss.item(),
            category_loss=category_loss.item(),
            mask_loss=mask_loss.item(),
        )


class ValidationStep:
    def __init__(
        self,
        criterion: Criterion,
        model: Solo,
        batch_adaptor: BatchAdaptor,
        to_masks: ToMasks,
    ) -> None:
        self.criterion = criterion
        self.model = model
        self.bath_adaptor = batch_adaptor
        self.to_masks = to_masks

    @torch.no_grad()
    def __call__(
        self,
        batch: Batch,
        on_end: Optional[Callable[[list[Tensor], list[Tensor]], Any]] = None,
    ) -> dict[str, float]:  # mask_batch, label_batch, logs
        self.model.eval()
        images, gt_mask_batch, gt_label_batch = batch
        gt_category_grids, mask_index = self.bath_adaptor(
            mask_batch=gt_mask_batch, label_batch=gt_label_batch
        )
        pred_category_grids, pred_all_masks = self.model(images)
        loss, category_loss, mask_loss = self.criterion(
            (
                pred_category_grids,
                pred_all_masks,
            ),
            (
                gt_category_grids,
                gt_mask_batch,
                mask_index,
            ),
        )
        pred_mask_batch, pred_label_batch = self.to_masks(
            pred_category_grids, pred_all_masks
        )
        if on_end is not None:
            on_end(pred_mask_batch, gt_mask_batch)
        return dict(
            loss=loss.item(),
            category_loss=category_loss.item(),
            mask_loss=mask_loss.item(),
        )
