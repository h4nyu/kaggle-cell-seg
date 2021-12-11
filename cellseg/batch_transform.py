import torch
import torch.nn.functional as F
import random
from cellseg.utils import round_to, log_rand, fix_bboxes
import math


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

        if random.random() >= self.p:
            return batch

        image_batch = batch["image"]
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
        new_batch["image"] = new_image_batch

        if "other_image" in batch:
            other_image_batch = batch["other_image"]
            new_other_image_batch = F.interpolate(
                other_image_batch,
                (new_height, new_width),
                mode="bilinear",
                align_corners=False,
            )
            new_batch["other_image"] = new_other_image_batch

        if "bboxes" in batch:
            bboxes_batch = batch["bboxes"]
            scale = torch.tensor([scale_x, scale_y] * 2).to(bboxes_batch[0])
            new_bboxes_batch = [fix_bboxes(bboxes) * scale for bboxes in bboxes_batch]
            new_batch["bboxes"] = new_bboxes_batch

        if "masks" in batch:
            masks_batch = batch["masks"]
            new_masks_batch = [
                F.interpolate(
                    masks.float().unsqueeze(0),
                    (new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                for masks in masks_batch
            ]
            new_batch["masks"] = new_masks_batch

        return new_batch
