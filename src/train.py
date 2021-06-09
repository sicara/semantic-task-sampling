from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers import SemanticTaskSampler
from easyfsl.methods import PrototypicalNetworks


@click.option(
    "--specs-dir",
    help="Where to find the dataset specs files",
    type=Path,
    required=True,
)
@click.option(
    "--distances-dir",
    help="Where to find class-distances matrix",
    type=Path,
    required=True,
)
@click.option(
    "--output-model",
    help="Where to dump the archive containing trained model weights",
    type=Path,
    required=True,
)
@click.command()
def main(specs_dir: Path, distances_dir: Path, output_model: Path):
    logger.info("Fetching training data...")
    train_set = EasySet(specs_file=specs_dir / "train.json", training=True)
    train_sampler = SemanticTaskSampler(
        train_set,
        n_way=5,
        n_shot=5,
        n_query=10,
        n_tasks=20,
        alpha=0.5,
        semantic_distances_csv=Path(distances_dir / "train.csv"),
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    logger.info("Building model...")
    convolutional_network = resnet18(pretrained=False)
    convolutional_network.fc = nn.Flatten()
    model = PrototypicalNetworks(convolutional_network).cuda()

    optimizer = Adam(params=model.parameters())

    logger.info("Starting training...")
    model.fit_multiple_epochs(train_loader, optimizer, n_epochs=2)

    torch.save(model.state_dict(), output_model)
    logger.info(f"Trained model weights dumped at {output_model}")


if __name__ == "__main__":
    main()
