import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .heads import Head
from typing import Protocol, TypedDict, Optional, Callable, Any
from cellseg.loss import FocalLoss, DiceLoss
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import masks_to_boxes, box_convert
from .util import grid, draw_save
from .backbones import FPNLike


Batch = tuple[Tensor, list[Tensor], list[Tensor]]  # id, images, mask_batch, label_batch


class ToCategoryGrid:
    def __init__(
        self,
        num_classes: int,
        grid_size: int,
    ) -> None:
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.to_index = CentersToGridIndex(self.grid_size)

    @torch.no_grad()
    def __call__(
        self,
        centers: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor]:  # category_grid, matched
        device = centers.device
        dtype = centers.dtype
        cagetory_grid = torch.zeros(
            self.num_classes, self.grid_size, self.grid_size, dtype=dtype
        ).to(device)
        if len(centers) == 0:
            matched = torch.zeros(0, 2).long().to(device)
            return cagetory_grid, matched
        center_index = torch.arange(len(centers)).to(device)
        mask_index = self.to_index(centers)
        flattend = cagetory_grid.view(-1)
        index = labels * self.grid_size ** 2 + mask_index
        flattend[index.long()] = 1
        cagetory_grid = flattend.view(self.num_classes, self.grid_size, self.grid_size)
        matched = torch.stack([mask_index, center_index], dim=1)
        return cagetory_grid, matched


class MasksToCenters:
    @torch.no_grad()
    def __call__(
        self,
        masks: Tensor,
    ) -> Tensor:
        boxes = masks_to_boxes(masks)
        cxcy = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")[:, :2]
        return cxcy


class CentersToGridIndex:
    def __init__(
        self,
        grid_size: int,
    ) -> None:
        self.grid_size = grid_size

    @torch.no_grad()
    def __call__(
        self,
        centers: Tensor,
    ) -> Tensor:
        device = centers.device
        if len(centers) == 0:
            return torch.zeros((0, 2)).long().to(device)
        return centers[:, 1].long() * self.grid_size + centers[:, 0].long()


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

    @torch.no_grad()
    def __call__(
        self,
        mask_batch: list[Tensor],
        label_batch: list[Tensor],
    ) -> tuple[
        Tensor, list[Tensor]
    ]:  # category_grids, list of mask_index, list of labels
        mask_index_batch: list[Tensor] = []
        cate_grids: list[Tensor] = []
        for masks, labels in zip(mask_batch, label_batch):
            scaled_centers = self.masks_to_centers(masks) * self.scale
            category_grid, mask_index = self.to_category_grid(
                centers=scaled_centers,
                labels=labels,
            )
            cate_grids.append(category_grid)
            mask_index_batch.append(mask_index)
        category_grids = torch.stack(cate_grids)
        return category_grids, mask_index_batch


class Criterion:
    def __init__(
        self,
        mask_weight: float = 1.0,
        category_weight: float = 1.0,
    ) -> None:
        self.category_loss = FocalLoss()
        self.mask_loss = FocalLoss()
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
        category_loss = self.category_loss(pred_category_grids, gt_category_grids)
        mask_loss = torch.tensor(0.0).to(device)
        count = 0
        for gt_masks, mask_index, pred_masks in zip(
            gt_mask_batch, mask_index_batch, all_masks
        ):
            if len(mask_index) > 0:
                mask_loss += self.mask_loss(
                    pred_masks[mask_index[:, 0]], gt_masks[mask_index[:, 1]]
                )
                count += 1
        mask_loss = mask_loss / count
        loss = self.category_weight * category_loss + self.mask_weight * mask_loss
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
            use_cord=False,
        )

        self.mask_head = Head(
            hidden_channels=hidden_channels,
            num_classes=grid_size ** 2,
            channels=backbone.channels[mask_feat_range[0] : mask_feat_range[1]],
            reductions=backbone.reductions[mask_feat_range[0] : mask_feat_range[1]],
            use_cord=True,
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


class MatrixNms:
    def __init__(self, kernel: str = "gaussian", sigma: float = 2.0) -> None:
        self.kernel = kernel
        self.sigma = sigma

    def __call__(
        self,
        seg_masks: Tensor,
        cate_labels: Tensor,
        cate_scores: Tensor,
        sum_masks: Optional[Tensor] = None,
    ) -> Tensor:
        """Matrix NMS for multi-class masks.
        Args:
            seg_masks (Tensor): shape (n, h, w)
            cate_labels (Tensor): shape (n), mask labels in descending order
            cate_scores (Tensor): shape (n), mask scores in descending order
            kernel (str):  'linear' or 'gauss'
            sigma (float): std in gaussian method
            sum_masks (Tensor): The sum of seg_masks
        Returns:
            Tensor: cate_scores_update, tensors of shape (n)
        """
        n_samples = len(seg_masks)
        if n_samples == 0:
            return seg_masks
        if sum_masks is None:
            sum_masks = seg_masks.sum((1, 2)).float()
        seg_masks = seg_masks.reshape(n_samples, -1).float()
        # inter.
        inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
        # union.
        sum_masks_x = sum_masks.expand(n_samples, n_samples)
        # iou.
        iou_matrix = (
            inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)
        ).triu(diagonal=1)
        # label_specific matrix.
        cate_labels_x = cate_labels.expand(n_samples, n_samples)
        label_matrix = (
            (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)
        )

        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == "gaussian":
            decay_matrix = torch.exp(-1 * self.sigma * (decay_iou ** 2))
            compensate_matrix = torch.exp(-1 * self.sigma * (compensate_iou ** 2))
            decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
        else:
            decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
            decay_coefficient, _ = decay_matrix.min(0)

        # update the score.
        cate_scores_update = cate_scores * decay_coefficient
        return cate_scores_update


class ToMasks:
    def __init__(
        self,
        category_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        kernel_size: int = 3,
        use_nms: bool = False,
    ) -> None:
        self.category_threshold = category_threshold
        self.mask_threshold = mask_threshold
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
        self.use_nms = use_nms
        self.nms = MatrixNms()

    @torch.no_grad()
    def __call__(
        self, category_grids: Tensor, all_masks: Tensor
    ) -> tuple[
        list[Tensor], list[Tensor], list[Tensor]
    ]:  # mask_batch, label_batch, mask_indecies
        batch_size = category_grids.shape[0]
        grid_size = category_grids.shape[2]
        category_grids = category_grids * (
                (self.max_pool(category_grids) == category_grids)
                & (category_grids > self.category_threshold)
                )

        to_index = CentersToGridIndex(grid_size=grid_size)
        all_masks = all_masks > self.mask_threshold
        (
            batch_indecies,
            all_labels,
            cy,
            cx,
        ) = category_grids.nonzero(as_tuple=True)
        mask_indecies = to_index(torch.stack([cx, cy], dim=1))
        mask_batch: list[Tensor] = []
        label_batch: list[Tensor] = []
        score_batch: list[Tensor] = []
        for batch_idx in range(batch_size):
            filterd = batch_indecies == batch_idx
            if len(filterd) == 0:
                mask_batch.append(
                    torch.zeros((0, *all_masks.shape[2:]), dtype=all_masks.dtype)
                )
                label_batch.append(torch.zeros((0,), dtype=torch.long))
                score_batch.append(torch.zeros((0,), dtype=torch.float))
                continue
            masks = all_masks[batch_idx][mask_indecies[filterd]]
            empty_filter = masks.sum(dim=[1, 2]) > 0
            masks = masks[empty_filter]
            labels = all_labels[filterd][empty_filter]
            scores = category_grids[
                batch_idx, labels, cy[filterd][empty_filter], cy[filterd][empty_filter]
            ]
            if self.use_nms:
                scores = self.nms(masks, labels, scores)
                score_filter = scores > self.category_threshold
                labels = labels[score_filter]
                masks = masks[score_filter]
                scores = scores[score_filter]
            label_batch.append(labels)
            mask_batch.append(masks)
            score_batch.append(scores)
        return mask_batch, label_batch, score_batch


class TrainStep:
    def __init__(
        self,
        criterion: Criterion,
        model: Solo,
        optimizer: Any,
        batch_adaptor: BatchAdaptor,
        use_amp: bool = True,
        to_masks: Optional[ToMasks] = None,
    ) -> None:
        self.criterion = criterion
        self.model = model
        self.bath_adaptor = batch_adaptor
        self.optimizer = optimizer
        self.use_amp = use_amp
        self.scaler = GradScaler()
        self.to_masks = to_masks

    def __call__(self, batch: Batch) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        with autocast(enabled=self.use_amp):
            images, gt_mask_batch, gt_label_batch = batch
            gt_category_grids, mask_index_batch = self.bath_adaptor(
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
                    mask_index_batch,
                ),
            )
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.to_masks is not None:
            pred_mask_batch, _ = self.to_masks(pred_category_grids, pred_all_masks)
            draw_save(
                "/store/gt.png",
                images[0],
                gt_mask_batch[0],
            )
            draw_save(
                "/store/pred.png",
                images[0],
                pred_mask_batch[0],
            )

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
        use_amp: bool = True,
    ) -> None:
        self.criterion = criterion
        self.model = model
        self.batch_adaptor = batch_adaptor
        self.to_masks = to_masks
        self.use_amp = use_amp

    @torch.no_grad()
    def __call__(
        self,
        batch: Batch,
        on_end: Optional[Callable[[list[Tensor], list[Tensor]], Any]] = None,
    ) -> dict[str, float]:  # logs
        self.model.eval()
        with autocast(enabled=self.use_amp):
            images, gt_mask_batch, gt_label_batch = batch
            gt_category_grids, mask_index_batch = self.batch_adaptor(
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
                    mask_index_batch,
                ),
            )
            if on_end is not None:
                pred_mask_batch, pred_label_batch, score_batch = self.to_masks(
                    pred_category_grids, pred_all_masks
                )
                on_end(pred_mask_batch, gt_mask_batch)
        return dict(
            loss=loss.item(),
            category_loss=category_loss.item(),
            mask_loss=mask_loss.item(),
        )


class InferenceStep:
    def __init__(
        self,
        model: Solo,
        batch_adaptor: BatchAdaptor,
        to_masks: ToMasks,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.bath_adaptor = batch_adaptor
        self.to_masks = to_masks
        self.use_amp = use_amp

    @torch.no_grad()
    def __call__(
        self,
        batch: Batch,
    ) -> tuple[list[Tensor], list[Tensor]]:  # mask_batch, label_batch
        self.model.eval()
        with autocast(enabled=self.use_amp):
            images, gt_mask_batch, gt_label_batch = batch
            gt_category_grids, mask_index = self.bath_adaptor(
                mask_batch=gt_mask_batch, label_batch=gt_label_batch
            )
            pred_category_grids, pred_all_masks = self.model(images)
            pred_mask_batch, pred_label_batch, score_batch = self.to_masks(
                pred_category_grids, pred_all_masks
            )
            return pred_mask_batch, pred_label_batch
