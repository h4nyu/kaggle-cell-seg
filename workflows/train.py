import hydra
from torch import Tensor
from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger
from cellseg.solo import Solo, TrainStep, Loss
from cellseg.solo.adaptors import BatchAdaptor
from cellseg.backbones import EfficientNetFPN
from cellseg.dataset import get_fold_indices, CellTrainDataset, collate_fn
from torch import optim
from torch.utils.data import Subset, DataLoader


@hydra.main(config_path="/app/config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = getLogger(cfg.name)
    backbone = EfficientNetFPN(**cfg.backbone)
    model = Solo(**cfg.model, backbone=backbone)
    optimizer = optim.SGD(model.parameters(), **cfg.optimizer)
    loss = Loss()
    batch_adaptor = BatchAdaptor(
        num_classes=cfg.num_classes,
        grid_size=cfg.grid_size,
        original_size=cfg.original_size,
    )
    train_step = TrainStep(
        model=model,
        loss=loss,
        batch_adaptor=batch_adaptor,
    )
    dataset = CellTrainDataset(**cfg.dataset)
    train_indecies, validation_indecies = get_fold_indices(dataset, **cfg.fold)
    train_loader = DataLoader(
        Subset(dataset, train_indecies), collate_fn=collate_fn, **cfg.train_loader
    )
    val_loader = DataLoader(
        Subset(dataset, validation_indecies),
        collate_fn=collate_fn,
        **cfg.validation_loader
    )

    for epoch in range(cfg.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch, data in enumerate(train_loader, 0):
            print(batch)


if __name__ == "__main__":
    main()
