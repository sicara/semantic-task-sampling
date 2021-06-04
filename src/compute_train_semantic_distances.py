from pathlib import Path
import pandas as pd

from easyfsl.data_tools import EasySet

train_set = EasySet(specs_file="./data/tiered_imagenet/train.json", training=False)
train_set.labels = 10 * list(range(len(train_set.class_names)))

semantic_distances_df = pd.DataFrame(
    train_set.get_semantic_distance_matrix(
        train_set.get_semantic_dag(Path("data/tiered_imagenet/wordnet.is_a.txt"))
    )
)

semantic_distances_df.to_csv(
    "data/tiered_imagenet/train_semantic_distances.csv", index=False, header=False
)
