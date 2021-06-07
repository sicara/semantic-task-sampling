from pathlib import Path
import pandas as pd
from loguru import logger

from easyfsl.data_tools import EasySet, EasySemantics

logger.info("Creating dataset...")
train_set = EasySet(specs_file="data/tiered_imagenet/train.json", training=False)
semantic_tools = EasySemantics(train_set, Path("data/tiered_imagenet/wordnet.is_a.txt"))

logger.info("Computing semantic distances...")
semantic_distances_df = pd.DataFrame(semantic_tools.get_semantic_distance_matrix())

semantic_distances_df.to_csv(
    "data/tiered_imagenet/train_semantic_distances.csv", index=False, header=False
)
