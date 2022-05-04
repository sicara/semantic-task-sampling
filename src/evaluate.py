from pathlib import Path

import click
from loguru import logger

from src.easyfsl.data_tools import EasySet, DanishFungi
from src.easyfsl.data_tools.samplers import TestbedSampler
from src.utils import build_model, create_dataloader


@click.option(
    "--specs-dir",
    help="Where to find the dataset specs files",
    type=Path,
    required=True,
)
@click.option(
    "--testbed",
    help="Path to the CSV defining the testbed",
    type=Path,
    required=True,
)
@click.option(
    "--method",
    help="Few-Shot Method",
    type=str,
    default="PrototypicalNetworks",
)
@click.option(
    "--dataset",
    help="Dataset (fungi or tiered_imagenet)",
    type=str,
    default="tiered_imagenet",
)
@click.option(
    "--trained-model",
    help="Path to an archive containing trained model weights",
    type=Path,
    default=None,
)
@click.option(
    "--output-dir", help="Where to dump evaluation results", type=Path, required=True
)
@click.option(
    "--device",
    help="What device to train the model on",
    type=str,
    default="cuda",
)
@click.command()
def main(
    specs_dir: Path,
    testbed: Path,
    method: str,
    dataset: str,
    trained_model: Path,
    output_dir: Path,
    device: str,
):
    n_workers = 20

    logger.info("Fetching test data...")
    if dataset == "tiered_imagenet":
        test_set = EasySet(specs_file=specs_dir / "test.json", training=False)
    elif dataset == "fungi":
        test_set = DanishFungi()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    test_sampler = TestbedSampler(
        test_set,
        testbed,
    )
    test_loader = create_dataloader(test_set, test_sampler, n_workers)

    logger.info("Retrieving model...")
    model = build_model(
        dataset=dataset, device=device, pretrained_weights=trained_model, method=method
    )

    logger.info("Starting evaluation...")
    results = model.evaluate(test_loader)

    output_file = output_dir / f"raw_results{testbed.stem.lstrip('testbed')}.csv"
    results.to_csv(output_file)
    logger.info(f"Raw results dumped at {output_file}")


if __name__ == "__main__":
    main()
