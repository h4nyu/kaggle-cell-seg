import torch
import torch.nn.functional as F
import random
from torch import Tensor
from cellseg.utils import round_to, log_rand, fix_bboxes
import math


class BatchTile:
    @torch.no_grad()
    def __call__(
        self, batches: list[dict[str, torch.Tensor]]
    ) -> list[dict[str, Tensor]]:
        assert len(batches) == 4
        batch_size, _, rows, cols = batches[0]["images"].shape

        new_batch = batches[0].copy()

        new_batch["images"] = tile_images([b["images"] for b in batches])

        if "box_batch" in batches[0]:
            new_batch["box_batch"] = [
                tile_bboxes([b["box_batch"][idx] for b in batches], (cols, rows))
                for idx in range(batch_size)
            ]

        if "label_batch" in batches[0]:
            new_batch["label_batch"] = [
                torch.cat([b["labels"][idx] for b in batches])
                for idx in range(batch_size)
            ]

        if "masks" in batches[0]:
            new_batch["masks"] = [
                tile_masks([b["masks"][idx] for b in batches])
                for idx in range(batch_size)
            ]

        return new_batch


class RandomResize:
    def __init__(
        self,
        scale_lim: tuple[float, float] = (0.666, 1.0),
        isotoropic: bool = False,
        multiple: int = 32,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.scale_lim = scale_lim
        self.isotoropic = isotoropic
        self.multiple = multiple
        self.p = p

    @torch.no_grad()
    def __call__(self, batch: dict) -> dict:
        print(batch.keys())

        if random.random() >= self.p:
            return batch

        image_batch = batch["images"]
        height = image_batch.size(-2)
        width = image_batch.size(-1)

        scale_y_ = log_rand(*self.scale_lim)
        new_height = round_to(scale_y_ * height, self.multiple)
        scale_y = new_height / height
        if self.isotoropic:
            scale_x_ = scale_y_
        else:
            scale_x_ = log_rand(*self.scale_lim)
        new_width = round_to(scale_x_ * width, self.multiple)
        scale_x = new_width / width

        new_batch = batch.copy()

        new_image_batch = F.interpolate(
            image_batch, (new_height, new_width), mode="bilinear", align_corners=False
        )
        new_batch["images"] = new_image_batch

        if "other_image" in batch:
            other_image_batch = batch["other_image"]
            new_other_image_batch = F.interpolate(
                other_image_batch,
                (new_height, new_width),
                mode="bilinear",
                align_corners=False,
            )
            new_batch["other_image"] = new_other_image_batch

        if "box_batch" in batch:
            bboxes_batch = batch["box_batch"]
            scale = torch.tensor([scale_x, scale_y] * 2).to(bboxes_batch[0])
            new_bboxes_batch = [fix_bboxes(bboxes) * scale for bboxes in bboxes_batch]
            new_batch["box_batch"] = new_bboxes_batch

        if "mask_batch" in batch:
            masks_batch = batch["mask_batch"]
            new_masks_batch = [
                F.interpolate(
                    masks.float().unsqueeze(0),
                    (new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for masks in masks_batch
            ]
            new_batch["mask_batch"] = new_masks_batch

        return new_batch


class BatchRandomCrop:
    def __init__(self, size: tuple[int, int] = None) -> None:
        self.size = size

    @torch.no_grad()
    def __call__(self, batch: dict[str, torch.Tensor], size: tuple[int, int] = None):
        if size is None:
            patch_cols, patch_rows = self.size
        else:
            patch_cols, patch_rows = size

        batch_size, _, rows, cols = batch["images"].shape
        rois = torch.stack(
            [
                torch.randint(0, cols - patch_cols + 1, size=(batch_size,)),
                torch.randint(0, rows - patch_rows + 1, size=(batch_size,)),
            ],
            dim=-1,
        ).repeat(1, 2)
        rois[:, 2] += patch_cols
        rois[:, 3] += patch_rows

        new_batch = batch.copy()

        new_batch["images"] = torch.stack(
            [crop_image(image, roi) for image, roi in zip(batch["image"], rois)]
        )

        if "box_batch" in batch:
            new_batch["box_batch"] = [
                crop_bboxes(bboxes, roi)
                for bboxes, roi in zip(batch["box_batch"], rois)
            ]

        if "mask_batch" in batch:
            new_batch["mask_batch"] = [
                crop_image(masks, roi) for masks, roi in zip(batch["mask_batch"], rois)
            ]

        return new_batch


class BatchShuffle:
    @torch.no_grad()
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch_size = batch["images"].size(0)
        indices = torch.randperm(batch_size).tolist()

        new_batch = batch.copy()
        new_batch["images"] = batch["images"][indices]

        if "box_batch" in batch:
            bboxes_batch = batch["box_batch"]
            new_batch["box_batch"] = [bboxes_batch[idx] for idx in indices]

        if "label_batch" in batch:
            labels_batch = batch["label_batch"]
            new_batch["label_batch"] = [labels_batch[idx] for idx in indices]

        if "mask_batch" in batch:
            masks_batch = batch["mask_batch"]
            new_batch["mask_batch"] = [masks_batch[idx] for idx in indices]

        return new_batch


class Mosaic:
    def __init__(self, p: float = 1.0):
        self.p = p
        self.shuffle = BatchShuffle()
        self.random_crop = BatchRandomCrop()
        self.tile = BatchTile()
        self.clean = BatchClean()

    @torch.no_grad()
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if random.random() > self.p:
            return batch
        size = self._get_size(batch)
        batches = [self.shuffle(batch) for _ in range(4)]
        merged_batch = self.tile(batches)
        new_batch = self.random_crop(merged_batch, size)
        new_batch = self.clean(new_batch)
        return new_batch

    def _get_size(self, batch: dict[str, torch.Tensor]):
        rows, cols = batch["image"].shape[-2:]
        return (cols, rows)
