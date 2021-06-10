from functools import partial
from pathlib import Path

import click
from loguru import logger
from matplotlib import pyplot as plt
import pandas as pd

from src.utils import get_distance_std, get_median_distance, get_accuracies


@click.option(
    "--distances-dir",
    help="Where to find class-distances matrix",
    type=Path,
    required=True,
)
@click.option(
    "--metrics-dir",
    help="Where to find and dump evaluation metrics",
    type=Path,
    required=True,
)
@click.command()
def main(distances_dir: Path, metrics_dir: Path):
    results = pd.read_csv(metrics_dir / "raw_results.csv", index_col=0)
    distances = pd.read_csv(distances_dir / "test.csv", header=None).values

    statistics = (
        results.groupby("task_id")
        .true_label.unique()
        .apply(
            [
                partial(get_median_distance, distances=distances),
                partial(get_distance_std, distances=distances),
            ]
        )
        .join(get_accuracies(results))
    ).rename(
        columns={
            "get_median_distance": "median_class_distance",
            "get_distance_std": "std_class_distance",
        }
    )

    logger.info(
        f"Evaluation accuracy {100 * statistics['accuracy'].mean()}%"
        f" +- {100 * statistics['accuracy'].std()}%"
    )

    stats_file = metrics_dir / "task_performances.csv"
    statistics.to_csv(stats_file)
    logger.info(f"Task statistics dumped at {stats_file}")

    plot_file = metrics_dir / "accuracy_v_task_class_distance.png"
    statistics.plot(x="median_class_distance", y="accuracy", kind="scatter")
    plt.savefig(plot_file)
    logger.info(
        f"Accuracy as a function of median intra-task class distance dumped at {plot_file}"
    )


if __name__ == "__main__":
    main()
