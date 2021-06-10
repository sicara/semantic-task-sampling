import itertools
from pathlib import Path
from statistics import median, stdev
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers import (
    SemanticTaskSampler,
    AdaptiveTaskSampler,
    UniformTaskSampler,
)


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


def get_accuracies(results: pd.DataFrame) -> pd.Series:
    return (
        results.sort_values("score", ascending=False)
        .drop_duplicates(["task_id", "image_id"])
        .sort_values(["task_id", "image_id"])
        .reset_index(drop=True)
        .assign(accuracy=lambda df: df.true_label == df.predicted_label)
        .groupby("task_id")
        .accuracy.mean()
    )


def get_sampler(
    sampler: str,
    dataset: EasySet,
    distances_csv: Path = None,
    semantic_alpha: float = None,
    adaptive_forgetting: float = None,
    adaptive_hardness: float = None,
):
    common_args = {
        "dataset": dataset,
        "n_way": 5,
        "n_shot": 5,
        "n_query": 10,
        "n_tasks": 200,
    }
    if sampler == "semantic":
        if semantic_alpha is None or distances_csv is None:
            raise ValueError("Missing arguments for semantic sampler")
        return SemanticTaskSampler(
            alpha=semantic_alpha, semantic_distances_csv=distances_csv, **common_args
        )
    if sampler == "adaptive":
        if adaptive_forgetting is None or adaptive_hardness is None:
            raise ValueError("Missing arguments for adaptive sampler")
        return AdaptiveTaskSampler(
            forgetting=adaptive_forgetting, hardness=adaptive_hardness, **common_args
        )
    if sampler == "uniform":
        return UniformTaskSampler(**common_args)
    else:
        raise ValueError(f"Unknown sampler : {sampler}")
