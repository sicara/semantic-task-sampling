import json
from pathlib import Path
from typing import List, Union

from PIL import Image
import networkx as nx
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from easyfsl.data_tools.dag_utils import build_wordnet_dag, reduce_to_leaves

NORMALIZE_DEFAULT = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class EasySet(Dataset):
    """
    A ready-to-use dataset. Will work for any dataset where the images are
    grouped in directories by class. It expects a JSON file defining the
    classes and where to find them. It must have the following shape:
        {
            "class_names": [
                "class_1",
                "class_2"
            ],
            "class_roots": [
                "path/to/class_1_folder",
                "path/to/class_2_folder"
            ]
        }
    """

    def __init__(self, specs_file: Union[Path, str], image_size=224, training=False):
        """
        Args:
            specs_file: path to the JSON file
            image_size: images returned by the dataset will be square images of the given size
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip.
        """
        specs = self.load_specs(Path(specs_file))

        self.images, self.labels = self.list_data_instances(specs["class_roots"])

        self.class_names = specs["class_names"]

        self.transform = self.compose_transforms(image_size, training)

    @staticmethod
    def load_specs(specs_file: Path) -> dict:
        """
        Load specs from a JSON file.
        Args:
            specs_file: path to the JSON file

        Returns:
            dictionary contained in the JSON file

        Raises:
            ValueError: if specs_file is not a JSON, or if it is a JSON and the content is not
                of the expected shape.
        """

        if specs_file.suffix != ".json":
            raise ValueError("EasySet requires specs in a JSON file.")

        specs = json.load(open(specs_file, "r"))

        if "class_names" not in specs.keys() or "class_roots" not in specs.keys():
            raise ValueError(
                "EasySet requires specs in a JSON file with the keys class_names and class_roots."
            )

        if len(specs["class_names"]) != len(specs["class_roots"]):
            raise ValueError(
                "Number of class names does not match the number of class root directories."
            )

        return specs

    @staticmethod
    def compose_transforms(image_size: int, training: bool) -> transforms.Compose:
        """
        Create a composition of torchvision transformations, with some randomization if we are
            building a training set.
        Args:
            image_size: size of dataset images
            training: whether this is a training set or not

        Returns:
            compositions of torchvision transformations
        """
        return (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**NORMALIZE_DEFAULT),
                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(**NORMALIZE_DEFAULT),
                ]
            )
        )

    @staticmethod
    def list_data_instances(class_roots: List[str]) -> (List[str], List[int]):
        """
        Explore the directories specified in class_roots to find all data instances.
        Args:
            class_roots: each element is the path to the directory containing the elements
                of one class

        Returns:
            list of paths to the images, and a list of same length containing the integer label
                of each image
        """
        images = []
        labels = []
        for class_id, class_root in enumerate(class_roots):
            class_images = [
                str(image_path)
                for image_path in Path(class_root).glob("*")
                if image_path.is_file()
            ]
            images += class_images
            labels += len(class_images) * [class_id]

        return images, labels

    def __getitem__(self, item: int):
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id

        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
            The type of the image object depends of the output type of self.transform. By default
            it's a torch.Tensor, however you are free to define any function as self.transform, and
            therefore any type for the output image. For instance, if self.transform = lambda x: x,
            then the output image will be of type PIL.Image.Image.
        """
        # Some images of ILSVRC2015 are grayscale, so we convert everything to RGB for consistence.
        # If you want to work on grayscale images, use torch.transforms.Grayscale in your
        # transformation pipeline.
        img = self.transform(Image.open(self.images[item]).convert("RGB"))
        label = self.labels[item]

        return img, label

    def __len__(self) -> int:
        return len(self.labels)

    def get_semantic_dag(self, path_to_wordnet: Path) -> nx.DiGraph:
        """
        Get the Directed Acyclic Graph where the leafs are the classes of the dataset, and that
        defines the semantic hierarchy of classes following the Wordnet hierarchy.
        Args:
        path_to_wordnet: path to a txt file where each line contains an edge
            (parent class, child class) of the Wordnet hierarchy.
        Returns:
            the semantic directed acyclic graph
        """
        wordnet_dag = build_wordnet_dag(path_to_wordnet)

        return reduce_to_leaves(wordnet_dag, self.class_names)

    def get_semantic_distance(
        self, semantic_dag: nx.DiGraph, class_a: int, class_b: int
    ) -> float:
        """
        Compute the Jiang and Conrath semantic distance between two classes of the dataset.
        It is defined as 2 log(C) - (log(A) + log(B)) where A (resp. B) is the number of images
        with label class_a (resp. class_b) in the dataset, and C is the number of images which label
        is a descendant of the lowest common ancestor of class_a and class_b in the semantic DAG.
        Note that the function is symmetric between class_a and class_b.
        Args:
            semantic_dag: Directed Acyclic Graph (DAG) where the leaves are the class of the dataset
            class_a: first class
            class_b: second class
        Returns:
            the Jiang and COnrath distance between class_a and class_b
        """
        if class_a == class_b:
            return 0.0

        lowest_common_ancestor = nx.lowest_common_ancestor(
            semantic_dag, self.class_names[class_a], self.class_names[class_b]
        )
        spanned = [
            node
            for node in nx.algorithms.dag.descendants(
                semantic_dag, lowest_common_ancestor
            )
            if semantic_dag.out_degree[node] == 0
        ]

        population_common_ancestor = sum(
            [self.labels.count(self.class_names.index(leave)) for leave in spanned]
        )

        population_a = self.labels.count(class_a)
        population_b = self.labels.count(class_b)

        return 2 * np.log(population_common_ancestor) - (
            np.log(population_a) + np.log(population_b)
        )

    def get_semantic_distance_matrix(self, semantic_dag: nx.DiGraph) -> np.ndarray:
        """
        Compute the Jiang and Conrath semantic distance between all classes of the dataset.
        The distance between class_a and class_b is defined as 2 log(C) - (log(A) + log(B)) where
        A (resp. B) is the number of images with label class_a (resp. class_b) in the dataset,
        and C is the number of images which label is a descendant of the lowest common ancestor of
        class_a and class_b in the semantic DAG.
        Args:
            semantic_dag: Directed Acyclic Graph (DAG) where the leaves are the class of the dataset
        Returns:
            symmetric square matrix of floats. Value at (i,j) is the semantic distance between
                classes i and j
        """
        distances = np.zeros((len(self.class_names), len(self.class_names)))

        for class_a in tqdm(range(len(self.class_names)), unit="classes"):
            for class_b in range(class_a, len(self.class_names)):
                distances[class_a, class_b] = self.get_semantic_distance(
                    semantic_dag, class_a, class_b
                )

        return distances + distances.T
