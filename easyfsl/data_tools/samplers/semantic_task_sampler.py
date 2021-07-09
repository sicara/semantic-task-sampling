import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from easyfsl.data_tools.samplers import AbstractTaskSampler
from easyfsl.data_tools.samplers.utils import sample_label_from_potential
from easyfsl.utils import fill_diagonal


class SemanticTaskSampler(AbstractTaskSampler):
    """
    Implements Semantic Task Sampling.
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.

    Classes are sampled so that classes that are semantically close have a higher probability of
    being sampled in a same task.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        n_query: int,
        n_tasks: int,
        semantic_distances_csv: Path,
        alpha: float,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
            semantic_distances_csv: path to a csv file containing pair-wise semantic distances
                between classes
            alpha: float factor weighting the importance of semantic distances in the sampling
        """
        super().__init__(
            dataset=dataset,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_tasks=n_tasks,
        )

        self.distances = torch.tensor(
            pd.read_csv(semantic_distances_csv, header=None).values
        )

        self.potential_matrix = fill_diagonal(torch.exp(-alpha * self.distances), 0)

    def _sample_labels(self) -> torch.Tensor:
        """
        Sample a first label uniformly, then sample other labels with a probability proportional to
        their potential given previously sampled labels, in a greedy fashion.
        Returns:
            1-dim tensor of sampled labels
        """
        to_yield = random.sample(self.items_per_label.keys(), 1)

        potential = self.potential_matrix[to_yield[0]]

        for _ in range(1, self.n_way):
            to_yield.append(sample_label_from_potential(potential))
            potential = potential * self.potential_matrix[to_yield[-1]]

        # pylint: disable=not-callable
        return torch.tensor(to_yield)
        # pylint: enable=not-callable
