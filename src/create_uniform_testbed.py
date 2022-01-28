import random
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from loguru import logger

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers import UniformTaskSampler
from src.utils import get_pseudo_variance, save_tasks_plots

DISTANCES_DIR = Path("data/tiered_imagenet/distances")
SPECS_FILE = Path("data/tiered_imagenet/specs/test.json")
N_TASKS = 8000
N_WAY = 5
N_SHOT = 5
N_QUERY = 10


@click.option(
    "--n-tasks",
    help="Number of tasks to sample",
    type=int,
    default=5000,
)
@click.option(
    "--n-way",
    help="Number of classes in each task",
    type=int,
    default=N_WAY,
)
@click.option(
    "--n-shot",
    help="Number of support images per class",
    type=int,
    default=N_SHOT,
)
@click.option(
    "--n-query",
    help="Number of query images per class",
    type=int,
    default=N_QUERY,
)
@click.option(
    "--distances-csv",
    help="Path to the csv containing the distance matrix",
    type=Path,
    default=DISTANCES_DIR / "test.csv",
)
@click.option(
    "--specs-json",
    help="Path to the JSON containing the specs of the test set",
    type=Path,
    default=SPECS_FILE,
)
@click.option(
    "--seed",
    help="Random seed.",
    type=int,
    default=0,
)
@click.option(
    "--out-file",
    help="Path to the csv where the test bed will be dumped",
    type=Path,
    required=True,
)
@click.command()
def main(
    n_tasks: int,
    n_way: int,
    n_shot: int,
    n_query: int,
    distances_csv: Path,
    specs_json: Path,
    seed: int,
    out_file: Path,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    distances = pd.read_csv(distances_csv, header=None).values

    logger.info(f"Sampling classes for {n_tasks} tasks...")
    test_set = EasySet(specs_json)
    test_sampler = UniformTaskSampler(
        test_set, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_tasks
    )

    logger.info("Sampling testbed ...")
    tasks_with_samples = pd.concat(
        [
            pd.DataFrame(
                {
                    "task": task,
                    "variance": get_pseudo_variance(
                        list(set([test_set.labels[image_id] for image_id in items])),
                        distances,
                    ),
                    "labels": [test_set.labels[image_id] for image_id in items],
                    "image_id": items,
                    "support": n_way * (n_query * [False] + n_shot * [True]),
                }
            )
            for task, items in enumerate(test_sampler)
        ]
    )

    save_tasks_plots(
        tasks_with_samples[["task", "variance", "labels"]].drop_duplicates(), out_file
    )
    tasks_with_samples.to_csv(out_file)
    logger.info(f"Testbed dumped to {out_file}")


if __name__ == "__main__":
    main()