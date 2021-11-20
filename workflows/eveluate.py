import torch
import hydra
from torch import Tensor
from omegaconf import DictConfig
import os
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger
import torch_optimizer as optim
from cellseg.solo import (
    Solo,
    TrainStep,
    Criterion,
    ValidationStep,
    ToMasks,
    InferenceStep,
)
from cellseg.solo.adaptors import BatchAdaptor
from cellseg.metrics import MaskAP
from cellseg.backbones import EfficientNetFPN
from cellseg.util import seed_everything, Checkpoint, MeanReduceDict
from cellseg.data import (
    ToDevice,
    get_fold_indices,
    CellTrainDataset,
    collate_fn,
    TrainTranform,
    Tranform,
    draw_save,
)
from torch.utils.data import Subset, DataLoader
from pathlib import Path


@hydra.main(config_path="/app/config", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = getLogger(cfg.name)
    backbone = EfficientNetFPN(**cfg.backbone)
    checkpoint = Checkpoint[Solo](
        root_path=os.path.join(cfg.data.root_path, f"{cfg.name}"),
        default_score=float("inf"),
    )
    model = Solo(**cfg.model, backbone=backbone)
    model, score = checkpoint.load_if_exists(model)
    model = model.to(cfg.device)
    batch_adaptor = BatchAdaptor(
        num_classes=cfg.num_classes,
        grid_size=cfg.model.grid_size,
        original_size=cfg.original_size,
    )
    to_masks = ToMasks(**cfg.to_masks)
    inference_step = InferenceStep(
        model=model,
        to_masks=to_masks,
        batch_adaptor=batch_adaptor,
        use_amp=cfg.use_amp,
    )
    to_masks = ToMasks(**cfg.to_masks)
    to_device = ToDevice(cfg.device)
    dataset = CellTrainDataset(
        **cfg.dataset,
        transform=Tranform(
            original_size=cfg.original_size,
        ),
    )
    loader = DataLoader(
        Subset(dataset, indices=list(range(10))),
        collate_fn=collate_fn,
        **cfg.validation_loader,
    )

    idx = 0
    mask_ap = MaskAP()
    for batch in loader:
        batch = to_device(*batch)
        images, gt_mask_batch, _ = batch
        mask_batch, _ = inference_step(batch)
        for image, masks, gt_masks in zip(images, mask_batch, gt_mask_batch):
            pred_count = masks.shape[0]
            gt_count = gt_masks.shape[0]
            logger.info(f"{idx=} {pred_count=} {gt_count=}")
            mask_ap.accumulate(masks, gt_masks)
            draw_save(
                os.path.join("/store", cfg.name, f"{idx}_pred.png"),
                image,
                masks,
            )
            draw_save(
                os.path.join("/store", cfg.name, f"{idx}_gt.png"),
                image,
                gt_masks,
            )
            idx += 1
    score = mask_ap.value
    logger.info(f"{score=}")


if __name__ == "__main__":
    main()
