import hydra
from torch import Tensor
from omegaconf import DictConfig
import os
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger, FileHandler
# import torch.optim as optim
import torch_optimizer as optim
from cellseg.yolox import (
    MaskYolo,
    Criterion,
    TrainStep,
    ValidationStep,
    InferenceStep,
)
from cellseg.assign import SimOTA
from cellseg.metrics import MaskAP
from cellseg.backbones import EfficientNetFPN
from cellseg.utils import seed_everything, Checkpoint, MeanReduceDict, ToDevice
from cellseg.data import (
    get_fold_indices,
    CellTrainDataset,
    collate_fn,
    TrainTranform,
    Tranform,
)
from cellseg.necks import CSPNeck
from torch.utils.data import Subset, DataLoader
from pathlib import Path


@hydra.main(config_path="/app/config", config_name="mask_yolo")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    logger = getLogger(cfg.name)
    Path(os.path.join(cfg.data.root_path, f"{cfg.name}")).mkdir(exist_ok=True)
    logger.addHandler(FileHandler(os.path.join(cfg.root_path, cfg.name, "train.log")))
    backbone = EfficientNetFPN(**cfg.backbone)
    checkpoint = Checkpoint[MaskYolo](
        root_path=os.path.join(cfg.data.root_path, f"{cfg.name}"),
        default_score=0.0,
    )
    neck = CSPNeck(
        in_channels=backbone.out_channels,
        out_channels=backbone.out_channels,
        strides=backbone.strides,
    )
    model = MaskYolo(
        backbone=backbone,
        neck=neck,
        mask_size=cfg.mask_size,
        box_feat_range=cfg.box_feat_range,
        mask_feat_range=cfg.mask_feat_range,
        num_classes=cfg.num_classes,
        patch_size=cfg.patch_size,
        score_threshold=cfg.score_threshold,
        mask_threshold=cfg.mask_threshold,
        box_iou_threshold=cfg.box_iou_threshold,
    )
    model, score = checkpoint.load_if_exists(model)
    model = model.to(cfg.device)
    assign = SimOTA(**cfg.assign)
    criterion = Criterion(model=model, assign=assign, **cfg.criterion)
    optimizer = optim.AdaBound(model.parameters(), **cfg.optimizer)
    train_step = TrainStep(
        optimizer=optimizer,
        criterion=criterion,
        use_amp=cfg.use_amp,
    )
    validation_step = ValidationStep(
        criterion=criterion,
    )
    train_dataset = CellTrainDataset(
        **cfg.dataset,
        transform=TrainTranform(
            size=cfg.patch_size,
        ),
    )
    val_dataset = CellTrainDataset(
        **cfg.dataset,
        transform=Tranform(
            size=cfg.patch_size,
        ),
    )
    train_indecies, validation_indecies = get_fold_indices(train_dataset, **cfg.fold)
    train_loader = DataLoader(
        Subset(train_dataset, train_indecies), collate_fn=collate_fn, **cfg.train_loader
    )
    val_loader = DataLoader(
        Subset(val_dataset, validation_indecies),
        collate_fn=collate_fn,
        **cfg.validation_loader,
    )
    to_device = ToDevice(cfg.device)

    for _ in range(cfg.num_epochs):
        train_reduer = MeanReduceDict(keys=cfg.log_keys)
        for batch in train_loader:
            batch = to_device(*batch)
            train_log = train_step(batch)
            train_reduer.accumulate(train_log)
            logger.info(f"train batch {train_log} ")
        logger.info(f"epoch train {train_reduer.value}")
        mask_ap = MaskAP(**cfg.mask_ap)
        for batch in val_loader:
            batch = to_device(*batch)
            validation_log = validation_step(batch, on_end=mask_ap.accumulate_batch)
        if score < mask_ap.value:
            score = checkpoint.save(model, mask_ap.value)
            logger.info(f"save checkpoint")
        logger.info(f"epoch eval {score=} {mask_ap.value=}")


if __name__ == "__main__":
    main()
