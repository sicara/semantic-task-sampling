import json
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from easyfsl.utils import get_accuracies


@click.option(
    "--testbed-spec",
    help="Which testbed",
    type=str,
    required=True,
)
@click.command()
def main(testbed_spec: str):
    testbed_path = Path("data/tiered_imagenet/testbeds") / f"testbed_{testbed_spec}.csv"
    results_path = (
        Path("data/tiered_imagenet/metrics") / f"raw_results_{testbed_spec}.csv"
    )
    metrics_dir = results_path.parent

    results = pd.read_csv(results_path, index_col=0)

    statistics = pd.concat(
        [
            pd.read_csv(testbed_path, index_col=0).groupby("task").variance.mean(),
            get_accuracies(results),
        ],
        axis=1,
    )

    stats_file = metrics_dir / f"task_performances_{testbed_spec}.csv"
    statistics.to_csv(stats_file)
    logger.info(f"Task statistics dumped at {stats_file}")

    metrics_json = metrics_dir / f"evaluation_metrics_{testbed_spec}.json"
    with open(metrics_json, "w") as file:
        json.dump(
            {
                "accuracy": statistics.accuracy.mean(),
                "std": statistics.accuracy.std(),
                "first_quartile_acc": statistics.loc[
                    statistics.variance < statistics.variance.quantile(0.25)
                ].accuracy.mean(),
                "second_quartile_acc": statistics.loc[
                    statistics.variance.between(
                        statistics.variance.quantile(0.25),
                        statistics.variance.quantile(0.50),
                    )
                ].accuracy.mean(),
                "third_quartile_acc": statistics.loc[
                    statistics.variance.between(
                        statistics.variance.quantile(0.50),
                        statistics.variance.quantile(0.75),
                    )
                ].accuracy.mean(),
                "fourth_quartile_acc": statistics.loc[
                    statistics.variance.quantile(0.75) <= statistics.variance
                ].accuracy.mean(),
            },
            file,
            indent=4,
        )
    logger.info(f"Metrics dumped to {metrics_json}")


if __name__ == "__main__":
    main()
