import pickle
from pathlib import Path
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from src.easyfsl.data_tools import EasySet


class EasySetLight(EasySet):
    """ """

    def __init__(self, data_source: Union[Path, str]):
        """
        Args:
            data_source: path to a pickle file with a dict where each key is an image_id and each
                value is a tuple (label, class_name, image) and each image is a (width, height, 3) ndarray.
        """
        with open(data_source, "rb") as f:
            self.data = pickle.load(f)

        self.labels = [x[0] for x in self.data.values()]

        label_to_class = {}
        for label, class_name, _ in self.data.values():
            label_to_class[label] = class_name

        self.class_names = [label_to_class[x] for x in sorted(set(self.labels))]

    def __getitem__(self, item: int):
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id

        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
        """
        img = torch.tensor(self.data[item][2])
        label = self.labels[item]

        return img, label

    def __len__(self) -> int:
        return len(self.labels)

    def number_of_classes(self):
        return len(self.class_names)


def generate_light_easyset(dataset: EasySet, output_path: Path):
    """
    Generate a light version of an EasySet dataset.
    Args:
        dataset: EasySet dataset

    Returns:
        data that was dumped to output_file as a pickle
    """
    data = {}
    for image_id, (image, label) in tqdm(enumerate(dataset)):
        data[image_id] = (label, dataset.class_names[label], np.array(image))

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return data
