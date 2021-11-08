from cellseg.dataset import CellTrainDataset


def test_cell_dataset() -> None:
    dataset = CellTrainDataset()
    assert len(dataset) == 606
    print(dataset[0])
