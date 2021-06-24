from functools import partial
from pathlib import Path

import click
from loguru import logger
from matplotlib import pyplot as plt
import pandas as pd

from src.utils import get_distance_std, get_median_distance, get_accuracies


@click.option(
    "--testbed",
    help="Path to the CSV defining the testbed",
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
def main(testbed: Path, metrics_dir: Path):
    results = pd.read_csv(metrics_dir / "raw_results.csv", index_col=0)

    statistics = pd.concat(
        [
            pd.read_csv(testbed, index_col=0).groupby("task").variance.mean(),
            get_accuracies(results),
        ],
        axis=1,
    )

    logger.info(
        f"Evaluation accuracy {100 * statistics['accuracy'].mean()}%"
        f" +- {100 * statistics['accuracy'].std()}%"
    )

    stats_file = metrics_dir / "task_performances.csv"
    statistics.to_csv(stats_file)
    logger.info(f"Task statistics dumped at {stats_file}")

    plot_file = metrics_dir / "accuracy_v_variance.png"
    statistics.plot(x="variance", y="accuracy", kind="scatter")
    plt.savefig(plot_file)
    logger.info(f"Accuracy as a function of task pseudo-variance dumped at {plot_file}")


if __name__ == "__main__":
    main()
