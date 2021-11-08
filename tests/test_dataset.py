from cellseg.dataset import CellTrainDataset


def test_cell_dataset() -> None:
    dataset = CellTrainDataset()
    assert len(dataset) == 606
    sample = dataset[0]
    assert sample is not None
    assert sample["image"].shape == (1, 520, 704)
    assert sample["labels"].shape[0] == sample["masks"].shape[0]
