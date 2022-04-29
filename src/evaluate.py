from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.danish_fungi import DanishFungi
from easyfsl.data_tools.samplers.testbed_sampler import TestbedSampler
from easyfsl.methods import PrototypicalNetworks
from src.utils import build_model, create_dataloader, build_model_trained_on_imagenet


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
    trained_model: Path,
    output_dir: Path,
    device: str,
):
    n_workers = 20

    logger.info("Fetching test data...")
    # We use trained_model as a marker of whether we're using Fungi.
    # TODO: fix this, will cause issues
    if trained_model:
        test_set = EasySet(specs_file=specs_dir / "test.json", training=False)
    else:
        test_set = DanishFungi()
    test_sampler = TestbedSampler(
        test_set,
        testbed,
    )
    test_loader = create_dataloader(test_set, test_sampler, n_workers)

    logger.info("Retrieving model...")
    # If we specify trained weights, we build a ResNet12 with those weights.
    if trained_model:
        model = build_model(
            device=device, pretrained_weights=trained_model, method=method
        )
    # Otherwise, we build a ResNet18 with torch's weights pretrained on ImageNet
    # Sorry about this hidden parameter, TODO: fix it.
    else:
        model = build_model_trained_on_imagenet(device=device, method=method)

    logger.info("Starting evaluation...")
    results = model.evaluate(test_loader)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_file = output_dir / f"raw_results{testbed.stem.lstrip('testbed')}.csv"
    results.to_csv(output_file)
    logger.info(f"Raw results dumped at {output_file}")


if __name__ == "__main__":
    main()
