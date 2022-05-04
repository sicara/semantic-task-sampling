import json
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from easyfsl.utils import top_k_accuracies


@click.option(
    "--testbed-spec",
    help="Which testbed",
    type=str,
    required=True,
)
@click.option(
    "--top-k",
    help="What top-k accuracies to report (multiple coma-separated integers)",
    type=str,
    default="1",
)
@click.option(
    "--testbeds-dir",
    help="Which testbed directory",
    type=Path,
    default=Path("data/tiered_imagenet/testbeds"),
)
@click.option(
    "--metrics-dir",
    help="Which testbed directory",
    type=Path,
    default=Path("data/tiered_imagenet/metrics"),
)
@click.command()
def main(testbed_spec: str, top_k: str, testbeds_dir: Path, metrics_dir: Path):
    testbed_path = testbeds_dir / f"testbed_{testbed_spec}.csv"
    results_path = metrics_dir / f"raw_results_{testbed_spec}.csv"
    metrics_dir = results_path.parent
    top_k_list = list(map(int, top_k.split(",")))

    results = pd.read_csv(results_path, index_col=0)

    statistics = pd.concat(
        [
            pd.read_csv(testbed_path, index_col=0).groupby("task").variance.mean(),
            top_k_accuracies(results, k=top_k_list),
        ],
        axis=1,
    )

    stats_file = metrics_dir / f"task_performances_{testbed_spec}.csv"
    statistics.to_csv(stats_file)
    logger.info(f"Task statistics dumped at {stats_file}")

    metrics_json = metrics_dir / f"evaluation_metrics_{testbed_spec}.json"
    with open(metrics_json, "w") as file:
        json.dump(
            dict(
                {
                    f"top_{k_instance}_acc": statistics[f"top_{k_instance}"].mean()
                    for k_instance in top_k_list
                },
                **{
                    f"top_{k_instance}_std": statistics[f"top_{k_instance}"].std()
                    for k_instance in top_k_list
                },
                **{
                    f"top_{k_instance}_qrtl_1": statistics[f"top_{k_instance}"]
                    .loc[statistics.variance < statistics.variance.quantile(0.25)]
                    .mean()
                    for k_instance in top_k_list
                },
                **{
                    f"top_{k_instance}_qrtl_2": statistics[f"top_{k_instance}"]
                    .loc[
                        statistics.variance.between(
                            statistics.variance.quantile(0.25),
                            statistics.variance.quantile(0.50),
                        )
                    ]
                    .mean()
                    for k_instance in top_k_list
                },
                **{
                    f"top_{k_instance}_qrtl_3": statistics[f"top_{k_instance}"]
                    .loc[
                        statistics.variance.between(
                            statistics.variance.quantile(0.50),
                            statistics.variance.quantile(0.75),
                        )
                    ]
                    .mean()
                    for k_instance in top_k_list
                },
                **{
                    f"top_{k_instance}_qrtl_4": statistics[f"top_{k_instance}"]
                    .loc[statistics.variance.quantile(0.75) <= statistics.variance]
                    .mean()
                    for k_instance in top_k_list
                },
            ),
            file,
            indent=4,
        )
    logger.info(f"Metrics dumped to {metrics_json}")


if __name__ == "__main__":
    main()
