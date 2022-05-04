import itertools
from pydoc import locate

import random
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.easyfsl import EasySet
from src.easyfsl.data_tools.samplers import (
    AbstractTaskSampler,
)
from src.config import BACKBONES_PER_DATASET


def plot_dag(dag: nx.DiGraph):
    """
    Utility function to quickly draw a Directed Acyclic Graph.
    Root is at the top, leaves are on the bottom.
    Args:
        dag: input directed acyclic graph
    """
    pos = graphviz_layout(dag, prog="dot")
    nx.draw(dag, pos, with_labels=False, node_size=10, arrows=False)
    plt.show()


def get_median_distance(labels: List[int], distances: np.ndarray) -> float:
    """
    From a list of labels and a matrix of pair-wise distances, compute the median
    distance of all possible pairs from the list.
    Args:
        labels: integer labels in range(len(distances))
        distances: square symmetric matrix

    Returns:
        median distance
    """
    return median(
        [
            distances[label_a, label_b]
            for label_a, label_b in itertools.combinations(labels, 2)
        ]
    )


def get_distance_std(labels: List[int], distances: np.ndarray):
    """
    From a list of labels and a matrix of pair-wise distances, compute the standard deviation
    of distances of all possible pairs from the list.
    Args:
        labels: integer labels in range(len(distances))
        distances: square symmetric matrix

    Returns:
        median distance
    """
    return stdev(
        [
            distances[label_a, label_b]
            for label_a, label_b in itertools.combinations(labels, 2)
        ]
    )


def get_pseudo_variance(labels: List[int], distances: np.ndarray) -> float:
    """
    From a list of labels and a matrix of pair-wise distances, compute the pseudo-variance
    distance of all possible pairs from the list, i.e. the mean of all square distances.
    Args:
        labels: integer labels in range(len(distances))
        distances: square symmetric matrix

    Returns:
        pseudo-variance
    """
    return mean(
        [
            (distances[label_a, label_b] ** 2)
            for label_a, label_b in itertools.combinations(labels, 2)
        ]
    )


def set_random_seed(seed: int):
    """
    Set random, numpy and torch random seed, for reproducibility of the training
    Args:
        seed: defined random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed : {seed}")


def create_dataloader(dataset: EasySet, sampler: AbstractTaskSampler, n_workers: int):
    """
    Create a torch dataloader of tasks from the input dataset sampled according
    to the input tensor.
    Args:
        dataset: dataset from which to sample tasks
        sampler: task sampler, must implement an episodic_collate_fn method
        n_workers: number of workers of the dataloader

    Returns:
        a dataloader of tasks
    """
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )


def build_model(
    dataset: str,
    method: str,
    device: str,
    tb_writer: Optional[SummaryWriter] = None,
    pretrained_weights: Optional[Path] = None,
    trainable_backbone: bool = False,
):
    """
    Build a model and cast it on the appropriate device
    Args:
        dataset: dataset name: for our experiments we have selected one model for each dataset
        method: few-shot classification method
        device: device on which to put the model
        tb_writer: a tensorboard writer to log training events
        pretrained_weights: if you want to use pretrained_weights for the backbone
        trainable_backbone: whether to allow gradients through the backbone

    Returns:
        a few-shot learning model
    """
    convolutional_network = BACKBONES_PER_DATASET[dataset]

    if not trainable_backbone:
        convolutional_network.requires_grad_(False)

    method_class = locate(f"easyfsl.methods.{method}")
    model = method_class(
        backbone=convolutional_network,
        tensorboard_writer=tb_writer,
        device=device,
    ).to(device)

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights), strict=False)

    return model


def save_tasks_plots(tasks: pd.DataFrame, out_file: Path):
    """
    Plot two histograms:
        - label occurrences: to evaluate the balance between classes in the testbed.
        - pseudo-variances: to evaluate that the testbed covers quasi-evenly a wide range of
            pseudo-variances
    Args:
        tasks: dataframe of tasks with columns "task", "variance", "labels"
        out_file: where the testbed will be dumped. We will save the plots next to it.
    """

    tasks.groupby("task").variance.mean().hist().set_title(
        "histogram of pseudo-variances"
    )
    pseudo_variances_file = out_file.parent / f"{out_file.stem}_pv.png"
    plt.savefig(pseudo_variances_file)

    plt.clf()
    tasks.labels.value_counts().hist().set_title("histogram of label occurrences")
    occurrences_file = out_file.parent / f"{out_file.stem}_occ.png"
    plt.savefig(occurrences_file)

    logger.info(f"Histogram of pseudo-variances dumped to {pseudo_variances_file}")
    logger.info(f"Histogram of label occurrences dumped to {occurrences_file}")
