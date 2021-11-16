import hydra
from torch import Tensor
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Optional
from logging import getLogger
from cellseg.solo import Solo, TrainStep, Criterion
from cellseg.solo.adaptors import BatchAdaptor
from cellseg.backbones import EfficientNetFPN
from cellseg.dataset import get_fold_indices, CellTrainDataset, collate_fn, Tranform
from cellseg.data import ToDevice
from torch import optim
from torch.utils.data import Subset, DataLoader


@hydra.main(config_path="/app/config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = getLogger(cfg.name)
    backbone = EfficientNetFPN(**cfg.backbone)
    model = Solo(**cfg.model, backbone=backbone).to(cfg.device)
    criterion = Criterion()
    batch_adaptor = BatchAdaptor(
        num_classes=cfg.num_classes,
        grid_size=cfg.model.grid_size,
        original_size=cfg.original_size,
    )
    train_step = TrainStep(
        optimizer=optim.SGD(model.parameters(), **cfg.optimizer),
        model=model,
        criterion=criterion,
        batch_adaptor=batch_adaptor,
    )
    dataset = CellTrainDataset(
        **cfg.dataset,
        transform=Tranform(
            original_size=cfg.original_size,
        )
    )
    train_indecies, validation_indecies = get_fold_indices(dataset, **cfg.fold)
    train_loader = DataLoader(
        Subset(dataset, train_indecies), collate_fn=collate_fn, **cfg.train_loader
    )
    val_loader = DataLoader(
        Subset(dataset, validation_indecies),
        collate_fn=collate_fn,
        **cfg.validation_loader
    )
    to_device = ToDevice(cfg.device)

    for epoch in range(cfg.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            batch = to_device(*batch)
            train_log = train_step(batch)

        for batch_idx, batch in enumerate(val_loader):
            batch = to_device(*batch)
            validation_log = train_step(batch)


if __name__ == "__main__":
    main()
