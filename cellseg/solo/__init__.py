import torch
import torch.nn as nn
from torch import Tensor
from .heads import Head
from typing import Protocol
from cellseg.loss import SigmoidFocalLoss, DiceLoss
from .adaptors import BatchAdaptor


class FPNLike(Protocol):
    channels: list[int]
    reductions: list[int]

    def __call__(self, x: Tensor) -> list[Tensor]:
        ...


Batch = tuple[Tensor, list[Tensor], list[Tensor]]  # id, images, mask_batch, label_batch


class Loss:
    def __init__(
        self,
    ) -> None:
        self.category_loss = SigmoidFocalLoss()
        self.mask_loss = DiceLoss()

    def __call__(
        self,
        inputs: tuple[Tensor, Tensor],  # pred_cate_grids, all_masks
        targets: tuple[
            Tensor, list[Tensor], list[Tensor]
        ],  # gt_cate_grids, mask_batch, mask_index_batch
    ) -> Tensor:
        pred_category_grids, all_masks = inputs
        gt_category_grids, gt_mask_batch, mask_index_batch = targets
        category_loss = self.category_loss(
            inputs=pred_category_grids, targets=gt_category_grids
        )
        mask_loss = torch.tensor(0.0)
        for gt_masks, mask_index, pred_masks in zip(
            gt_mask_batch, mask_index_batch, all_masks
        ):
            filtered_masks = pred_masks[mask_index]
            mask_loss += self.mask_loss(inputs=filtered_masks, targets=gt_masks)
        loss = category_loss + mask_loss
        return loss


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
        mask_feats = features[self.mask_feat_range[0] : self.mask_feat_range[1]]
        masks = self.mask_head(mask_feats)
        return (category_grid, masks)


class TrainStep:
    def __init__(
        self,
        loss: Loss,
        model: Solo,
        batch_adaptor: BatchAdaptor,
    ) -> None:
        self.loss = loss
        self.model = model
        self.bath_adaptor = batch_adaptor

    def __call__(self, batch: Batch) -> Tensor:
        self.model.train()
        images, gt_mask_batch, gt_label_batch = batch
        gt_category_grids, mask_index = self.bath_adaptor(
            mask_batch=gt_mask_batch, label_batch=gt_label_batch
        )
        pred_category_grids, pred_all_masks = self.model(images)
        loss = self.loss(
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
        return loss


class ValidationStep:
    def __init__(
        self,
        loss: Loss,
        model: Solo,
        batch_adaptor: BatchAdaptor,
    ) -> None:
        self.loss = loss
        self.model = model
        self.bath_adaptor = batch_adaptor

    def __call__(self, batch: Batch) -> Tensor:
        self.model.eval()
        images, gt_mask_batch, gt_label_batch = batch
        gt_category_grids, mask_index = self.bath_adaptor(
            mask_batch=gt_mask_batch, label_batch=gt_label_batch
        )
        pred_category_grids, pred_all_masks = self.model(images)
        loss = self.loss(
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
        return loss
