from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers import UniformTaskSampler
from easyfsl.methods import PrototypicalNetworks


@click.option(
    "--specs-dir",
    help="Where to find the dataset specs files",
    type=Path,
    required=True,
)
@click.option(
    "--trained-model",
    help="Path to an archive containing trained model weights",
    type=Path,
    required=True,
)
@click.option(
    "--output-dir", help="Where to dump evaluation results", type=Path, required=True
)
@click.command()
def main(specs_dir: Path, trained_model: Path, output_dir: Path):
    logger.info("Fetching test data...")
    test_set = EasySet(specs_file=specs_dir / "test.json", training=False)
    test_sampler = UniformTaskSampler(
        test_set,
        n_way=5,
        n_shot=5,
        n_query=10,
        n_tasks=20,
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    logger.info("Retrieving model...")
    convolutional_network = resnet18(pretrained=False)
    convolutional_network.fc = nn.Flatten()
    model = PrototypicalNetworks(backbone=convolutional_network).cuda()
    model.load_state_dict(torch.load(trained_model))

    logger.info("Starting evaluation...")
    results = model.evaluate(test_loader)

    output_file = output_dir / "raw_results.csv"
    results.to_csv(output_file)
    logger.info(f"Raw results dumped at {output_file}")


if __name__ == "__main__":
    main()
