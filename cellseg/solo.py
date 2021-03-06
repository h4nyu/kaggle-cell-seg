import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .heads import Head
from typing import Protocol, TypedDict, Optional, Callable, Any
from cellseg.loss import FocalLoss, DiceLoss
from torch.cuda.amp import GradScaler, autocast
from torchvision.ops import masks_to_boxes, box_convert
from .utils import grid, draw_save, ToPatches, MergePatchedMasks
from .backbones import FPNLike
from .necks import NeckLike


Batch = tuple[Tensor, list[Tensor], list[Tensor]]  # id, images, mask_batch, label_batch


class ToCategoryGrid:
    def __init__(
        self,
        num_classes: int,
        grid_size: int,
        patch_size: int,
    ) -> None:
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.reduction = patch_size // grid_size
        self.patch_size = patch_size

    @torch.no_grad()
    def __call__(
        self,
        masks: Tensor,
        labels: Tensor,
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor
    ]:  # category_grid, size_grid, matched, pos_mask
        device = masks.device
        category_grid = torch.zeros(
            self.num_classes, self.grid_size, self.grid_size, dtype=torch.float
        ).to(device)
        size_grid = torch.zeros(
            4, self.grid_size, self.grid_size, dtype=torch.float
        ).to(device)
        matched = torch.zeros(0, 2).long().to(device)
        pos_mask = torch.zeros(1, self.grid_size, self.grid_size, dtype=torch.bool).to(
            device
        )
        if len(masks) == 0:
            return category_grid, size_grid, matched, pos_mask
        boxes = box_convert(masks_to_boxes(masks), in_fmt="xyxy", out_fmt="cxcywh")
        cxcy_index = (boxes[:, :2] / self.reduction).long()

        mask_index = torch.arange(len(boxes)).to(device)
        position_index = cxcy_index[:, 1] * self.grid_size + cxcy_index[:, 0]
        matched = torch.stack([position_index, mask_index], dim=1)

        category_flattend = category_grid.view(-1)
        index = (labels * self.grid_size ** 2 + position_index).long()
        category_flattend[index] = 1
        category_grid = category_flattend.view(
            self.num_classes, self.grid_size, self.grid_size
        )

        offset_and_wh = torch.cat(
            [
                boxes[:, 2:] / self.patch_size,
                (boxes[:, :2] / self.reduction) - cxcy_index,
            ],
            dim=1,
        )
        for r, index in zip(offset_and_wh, cxcy_index):
            size_grid[:, index[1], index[0]] = r
        pos_mask = (
            category_grid.sum(dim=0).bool().view(1, self.grid_size, self.grid_size)
        )
        return category_grid, size_grid, matched, pos_mask


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
        patch_size: int,
    ) -> None:
        self.grid_size = grid_size
        self.to_category_grid = ToCategoryGrid(
            grid_size=grid_size,
            num_classes=num_classes,
            patch_size=patch_size,
        )

    @torch.no_grad()
    def __call__(
        self,
        mask_batch: list[Tensor],
        label_batch: list[Tensor],
    ) -> tuple[
        Tensor, Tensor, list[Tensor], Tensor
    ]:  # category_grids, list of mask_index, list of labels, pos_masks
        mask_index_batch: list[Tensor] = []
        cate_grids: list[Tensor] = []
        size_grids: list[Tensor] = []
        pos_masks: list[Tensor] = []
        for masks, labels in zip(mask_batch, label_batch):
            category_grid, size_grid, mask_index, pos_mask = self.to_category_grid(
                masks=masks,
                labels=labels,
            )
            cate_grids.append(category_grid)
            size_grids.append(size_grid)
            mask_index_batch.append(mask_index)
            pos_masks.append(pos_mask)
        return (
            torch.stack(cate_grids),
            torch.stack(size_grids),
            mask_index_batch,
            torch.stack(pos_masks),
        )


class Criterion:
    def __init__(
        self,
        mask_weight: float = 1.0,
        category_weight: float = 1.0,
        size_weight: float = 1.0,
    ) -> None:
        self.category_loss = FocalLoss()
        self.mask_loss = FocalLoss()
        self.size_loss = nn.MSELoss()
        self.category_weight = category_weight
        self.mask_weight = mask_weight
        self.size_weight = size_weight

    def __call__(
        self,
        inputs: tuple[
            Tensor, Tensor, Tensor
        ],  # pred_cate_grids, pred_size_grid, all_masks
        targets: tuple[
            Tensor, Tensor, list[Tensor], list[Tensor], Tensor
        ],  # gt_cate_grids, gt_size_grid, ,mask_batch, mask_index_batch, pos_masks
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pred_category_grids, pred_size_grid, all_masks = inputs
        (
            gt_category_grids,
            gt_size_grid,
            gt_mask_batch,
            mask_index_batch,
            pos_masks,
        ) = targets
        device = pred_category_grids.device
        category_loss = self.category_loss(pred_category_grids, gt_category_grids)
        mask_loss = torch.tensor(0.0).to(device)
        size_loss = self.size_loss(
            pred_size_grid.masked_select(pos_masks),
            gt_size_grid.masked_select(pos_masks),
        )
        for gt_masks, mask_index, pred_masks in zip(
            gt_mask_batch, mask_index_batch, all_masks
        ):
            if len(mask_index) > 0:
                mask_loss += self.mask_loss(
                    pred_masks[mask_index[:, 0]], gt_masks[mask_index[:, 1]]
                )
        mask_loss = mask_loss / len(gt_mask_batch)
        loss = (
            self.category_weight * category_loss
            + self.mask_weight * mask_loss
            + self.size_weight * size_loss
        )
        return loss, category_loss, size_loss, mask_loss


class Solo(nn.Module):
    def __init__(
        self,
        backbone: FPNLike,
        neck: NeckLike,
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
        self.neck = neck
        self.grid_size = grid_size
        self.category_head = Head(
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            in_channels=backbone.out_channels[
                category_feat_range[0] : category_feat_range[1]
            ],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
        )

        self.size_head = Head(
            hidden_channels=hidden_channels,
            out_channels=4,
            in_channels=backbone.out_channels[
                category_feat_range[0] : category_feat_range[1]
            ],
            reductions=backbone.reductions[
                category_feat_range[0] : category_feat_range[1]
            ],
        )

        self.mask_head = Head(
            hidden_channels=hidden_channels,
            out_channels=grid_size ** 2,
            in_channels=backbone.out_channels[mask_feat_range[0] : mask_feat_range[1]],
            reductions=backbone.reductions[mask_feat_range[0] : mask_feat_range[1]],
            coord_level=len(mask_feat_range) - 1,
        )

    def forward(self, image_batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.backbone(image_batch)
        features = self.neck(features)
        category_feats = features[
            self.category_feat_range[0] : self.category_feat_range[1]
        ]
        category_grid = self.category_head(category_feats)
        size_grid = self.size_head(category_feats)
        mask_feats = features[self.mask_feat_range[0] : self.mask_feat_range[1]]
        masks = self.mask_head(mask_feats)
        return (category_grid, size_grid, masks)


class MatrixNms:
    def __init__(self, kernel: str = "gaussian", sigma: float = 3.0) -> None:
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
        patch_size: int,
        category_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        kernel_size: int = 3,
        use_nms: bool = False,
        use_crop: bool = False,
    ) -> None:
        self.patch_size = patch_size
        self.category_threshold = category_threshold
        self.mask_threshold = mask_threshold
        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
        self.use_nms = use_nms
        self.use_crop = use_crop
        self.nms = MatrixNms()

    @torch.no_grad()
    def __call__(
        self,
        category_grids: Tensor,
        size_grids: Tensor,
        all_masks: Tensor,
    ) -> tuple[
        list[Tensor], list[Tensor], list[Tensor]
    ]:  # mask_batch, label_batch, mask_indecies
        batch_size = category_grids.shape[0]
        grid_size = category_grids.shape[2]
        device = category_grids.device
        category_grids = category_grids * (
            (category_grids > self.category_threshold)
            & (self.max_pool(category_grids) == category_grids)
        )
        all_masks = all_masks > self.mask_threshold
        (
            batch_indecies,
            all_labels,
            all_cy,
            all_cx,
        ) = category_grids.nonzero(as_tuple=True)
        all_cxcy = torch.stack([all_cx, all_cy], dim=1)
        mask_indecies = all_cy * grid_size + all_cx
        mask_batch: list[Tensor] = []
        label_batch: list[Tensor] = []
        score_batch: list[Tensor] = []
        reduction = self.patch_size // grid_size
        for batch_idx in range(batch_size):
            filterd = batch_indecies == batch_idx
            if len(filterd) == 0:
                mask_batch.append(
                    torch.zeros((0, *all_masks.shape[2:]), dtype=all_masks.dtype)
                )
                label_batch.append(torch.zeros((0,), dtype=torch.long))
                score_batch.append(torch.zeros((0,), dtype=torch.float))
                continue

            labels = all_labels[filterd]
            cxcy = all_cxcy[filterd]
            scores = category_grids[batch_idx, labels, cxcy[:, 1], cxcy[:, 0]]
            box_sizes = (
                size_grids[batch_idx, :2, cxcy[:, 1], cxcy[:, 0]].t() * self.patch_size
            )
            cxcy_offsets = size_grids[batch_idx, 2:, cxcy[:, 1], cxcy[:, 0]].t()
            cxcywh = torch.cat([(cxcy + cxcy_offsets) * reduction, box_sizes], dim=1)
            boxes = (
                box_convert(cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
                .round()
                .long()
                .clip(min=0, max=self.patch_size - 1)
            )
            masks = all_masks[batch_idx][mask_indecies[filterd]]

            # crop masks by boxes
            if self.use_crop:
                for i, (m, b) in enumerate(zip(masks, boxes)):
                    crop_mask = torch.zeros(m.shape).to(device)
                    crop_mask[b[1] : b[3] + 1, b[0] : b[2] + 1] = 1
                    masks[i] = m * crop_mask

            empty_filter = masks.sum(dim=[1, 2]) > 0
            masks = masks[empty_filter]
            scores = scores[empty_filter]
            labels = labels[empty_filter]

            if self.use_nms:
                scores = self.nms(masks, labels, scores)
                score_filter = scores > self.category_threshold
                if len(score_filter) > 0:
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
        scheduler: Optional[Any] = None,
    ) -> None:
        self.criterion = criterion
        self.model = model
        self.bath_adaptor = batch_adaptor
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
                mask_index_batch,
                pos_masks,
            ) = self.bath_adaptor(mask_batch=gt_mask_batch, label_batch=gt_label_batch)
            pred_category_grids, pred_size_grids, pred_all_masks = self.model(images)
            loss, category_loss, size_loss, mask_loss = self.criterion(
                (
                    pred_category_grids,
                    pred_size_grids,
                    pred_all_masks,
                ),
                (
                    gt_category_grids,
                    gt_size_grids,
                    gt_mask_batch,
                    mask_index_batch,
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
                pred_category_grids, pred_size_grids, pred_all_masks
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
            mask_index_batch,
            pos_masks,
        ) = self.batch_adaptor(mask_batch=gt_mask_batch, label_batch=gt_label_batch)
        pred_category_grids, pred_size_grids, pred_all_masks = self.model(images)
        loss, category_loss, size_loss, mask_loss = self.criterion(
            (
                pred_category_grids,
                pred_size_grids,
                pred_all_masks,
            ),
            (
                gt_category_grids,
                gt_size_grids,
                gt_mask_batch,
                mask_index_batch,
                pos_masks,
            ),
        )
        if on_end is not None:
            pred_mask_batch, pred_label_batch, score_batch = self.to_masks(
                pred_category_grids, pred_size_grids, pred_all_masks
            )
            on_end(pred_mask_batch, gt_mask_batch)
        return dict(
            loss=loss.item(),
            category_loss=category_loss.item(),
            size_loss=size_loss.item(),
            mask_loss=mask_loss.item(),
        )


class InferenceStep:
    def __init__(
        self,
        model: Solo,
        batch_adaptor: BatchAdaptor,
        to_masks: ToMasks,
        use_amp: bool = True,
        to_patches: Optional[ToPatches] = None,
    ) -> None:
        self.model = model
        self.bath_adaptor = batch_adaptor
        self.to_masks = to_masks
        self.use_amp = use_amp

    @torch.no_grad()
    def __call__(
        self,
        images: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:  # mask_batch, label_batch
        self.model.eval()
        with autocast(enabled=self.use_amp):
            pred_category_grids, pred_size_grids, pred_all_masks = self.model(images)
            pred_mask_batch, pred_label_batch, score_batch = self.to_masks(
                pred_category_grids, pred_size_grids, pred_all_masks
            )
            return pred_mask_batch, pred_label_batch


class PatchInferenceStep:
    def __init__(
        self,
        model: Solo,
        batch_adaptor: BatchAdaptor,
        to_masks: ToMasks,
        patch_size: int,
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.bath_adaptor = batch_adaptor
        self.to_masks = to_masks
        self.use_amp = use_amp
        self.to_patches = ToPatches(patch_size)
        self.merge_masks = MergePatchedMasks(patch_size)

    @torch.no_grad()
    def __call__(
        self,
        images: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:  # mask_batch, label_batch
        self.model.eval()
        with autocast(enabled=self.use_amp):
            _, _, h, w = images.shape
            padded_images, patch_batch, patch_grid = self.to_patches(images)
            pred_mask_batch = []
            pred_label_batch = []
            for patches in patch_batch:
                pred_category_grids, pred_size_grids, pred_all_masks = self.model(
                    patches
                )
                patch_mask_batch, patch_label_batch, patch_score_batch = self.to_masks(
                    pred_category_grids, pred_size_grids, pred_all_masks
                )
                pred_masks = self.merge_masks(patch_mask_batch, patch_grid)[:, :h, :w]
                pred_mask_batch.append(pred_masks)
                pred_label_batch.append(torch.cat(patch_label_batch))
            return pred_mask_batch, pred_label_batch
