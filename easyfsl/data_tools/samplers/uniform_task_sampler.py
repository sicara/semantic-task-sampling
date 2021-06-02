import random

import torch

from easyfsl.data_tools.samplers import AbstractTaskSampler


class UniformTaskSampler(AbstractTaskSampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    self._sample_items_from_label(label)
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            )
