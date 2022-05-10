from pathlib import Path

import click
import pandas as pd
from loguru import logger

from src.easyfsl.data_tools import EasySemantics, EasySet, DanishFungi


@click.option(
    "--specs-dir",
    help="Where to find the dataset specs files",
    type=Path,
    required=True,
)
@click.option(
    "--output-dir",
    help="Where to dump class-distances matrix",
    type=Path,
    required=True,
)
@click.option(
    "--dataset", help="tiered_imagenet or fungi", type=str, default="tiered_imagenet"
)
@click.option(
    "--split",
    help="What split of the dataset to work on",
    type=click.Choice(["train", "val", "test"]),
    default="train",
)
@click.command()
def main(specs_dir: Path, output_dir: Path, dataset: str, split: str):

    logger.info("Creating dataset...")
    if dataset == "fungi":
        easy_set = DanishFungi()
        semantic_tools = EasySemantics(
            easy_set, specs_dir / "fungi_dag.json", is_fungi=True
        )
        output_file = output_dir / "distances.csv"
    else:
        easy_set = EasySet(specs_file=specs_dir / f"{split}.json", training=False)
        semantic_tools = EasySemantics(easy_set, Path(specs_dir / "wordnet.is_a.txt"))
        output_file = output_dir / f"{split}.csv"

    logger.info("Computing semantic distances...")
    semantic_distances_df = pd.DataFrame(semantic_tools.get_semantic_distance_matrix())

    semantic_distances_df.to_csv(output_file, index=False, header=False)


if __name__ == "__main__":
    main()
