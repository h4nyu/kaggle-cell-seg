import pandas as pd
from cellseg.config import TRAIN_FILE_PATH
import matplotlib.pyplot as plt


train_df = pd.read_csv(TRAIN_FILE_PATH)
id_groups = train_df.groupby("id")

image_cell_type_freq = id_groups["cell_type"].nunique()
print(f"{image_cell_type_freq=}")

count_of_multi_type = sum(image_cell_type_freq > 1)
print(f"{count_of_multi_type=}")

cell_type_counts = id_groups["id", "cell_type"].head(1).groupby("cell_type").count()

print(f"{cell_type_counts=}")
