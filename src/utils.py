import itertools
from statistics import median, stdev
from typing import List

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


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
