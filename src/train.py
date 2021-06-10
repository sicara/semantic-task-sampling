import pickle
from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.methods import PrototypicalNetworks
from src.utils import get_sampler

SAMPLERS = [
    "uniform",
    "adaptive",
    "semantic",
]


@click.option(
    "--sampler",
    help="How to sample training tasks",
    type=click.Choice(SAMPLERS),
    default="uniform",
)
@click.option(
    "--semantic-alpha",
    help="Weight of semantic distances for class sampling",
    type=float,
    default=0.5,
)
@click.option(
    "--adaptive-forgetting",
    help="Forgetting hyperparameter for adaptive sampling",
    type=float,
    default=0.5,
)
@click.option(
    "--adaptive-hardness",
    help="Hardness hyperparameter for adaptive sampling",
    type=float,
    default=0.5,
)
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
    "--metrics-dir",
    help="Where to find and dump evaluation metrics",
    type=Path,
    required=True,
)
@click.option(
    "--tb-log-dir",
    help="Where to dump tensorboard event files",
    type=Path,
    required=True,
)
@click.option(
    "--output-model",
    help="Where to dump the archive containing trained model weights",
    type=Path,
    required=True,
)
@click.option(
    "--device",
    help="What device to train the model on",
    type=str,
    default="cuda",
)
@click.command()
def main(
    sampler: str,
    semantic_alpha: float,
    adaptive_forgetting: float,
    adaptive_hardness: float,
    specs_dir: Path,
    distances_dir: Path,
    metrics_dir: Path,
    tb_log_dir: Path,
    output_model: Path,
    device: Path,
):
    logger.info("Fetching training data...")
    train_set = EasySet(specs_file=specs_dir / "train.json", training=True)
    train_sampler = get_sampler(
        sampler=sampler,
        dataset=train_set,
        distances_csv=distances_dir / "train.csv",
        semantic_alpha=semantic_alpha,
        adaptive_forgetting=adaptive_forgetting,
        adaptive_hardness=adaptive_hardness,
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    logger.info("Fetching validation data...")
    val_set = EasySet(specs_file=specs_dir / "val.json", training=True)
    val_sampler = get_sampler(
        sampler="uniform",
        dataset=val_set,
        distances_csv=distances_dir / "train.json",
        semantic_alpha=semantic_alpha,
        adaptive_forgetting=adaptive_forgetting,
        adaptive_hardness=adaptive_hardness,
    )
    val_loader = DataLoader(
        val_set,
        batch_sampler=val_sampler,
        num_workers=12,
        pin_memory=True,
        collate_fn=val_sampler.episodic_collate_fn,
    )

    tb_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building model...")
    convolutional_network = resnet18(pretrained=False)
    convolutional_network.fc = nn.Flatten()
    model = PrototypicalNetworks(
        backbone=convolutional_network,
        tensorboard_writer=SummaryWriter(log_dir=tb_log_dir),
        device=device,
    ).to(device)

    optimizer = Adam(params=model.parameters())

    logger.info("Starting training...")
    training_tasks_record = model.fit_multiple_epochs(
        train_loader,
        optimizer,
        n_epochs=2,
        val_loader=val_loader,
    )

    record_dump_path = metrics_dir / "training_tasks.pkl"
    pickle.dump(training_tasks_record, open(record_dump_path, "wb"))
    logger.info(f"Training tasks record dumped at {record_dump_path}")

    torch.save(model.state_dict(), output_model)
    logger.info(f"Trained model weights dumped at {output_model}")


if __name__ == "__main__":
    main()
