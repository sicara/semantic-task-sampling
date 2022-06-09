import pickle
from pathlib import Path
from typing import Union

import pandas as pd
import s3fs
from PIL import Image
from tqdm import tqdm

from src.easyfsl.data_tools import EasySet


class EasySetExpo(EasySet):
    """
    EasySet but we fetch the data from S3 and we don't do any preprocessing of images.
    """

    def __init__(
        self,
        specs_file: Union[Path, str],
        s3_root: str,
        local: bool = False,
        local_root: Path = None,
    ):
        """
        Args:
            specs_file: path to the JSON file needed by EasySet
            s3_root: s3://bucket/prefix/
            local: use local file system instead of S3
            local_root: only used if local=True
        """
        specs = self.load_specs(Path(specs_file))
        self.specs_file = specs_file

        self.class_names = specs["class_names"]

        self.fs = s3fs.S3FileSystem(anon=False)
        self.s3_root = s3_root

        self.local = local
        self.local_root = local_root

        if self.local:
            self.data = pd.read_csv(self.local_root / "test_images_list.csv")
        else:
            self.data = pd.read_csv(self.fs.open(f"{s3_root}test_images_list.csv"))

        self.images = self.data["image_name"].values.tolist()
        self.labels = self.data["label"].values.tolist()

    def __getitem__(self, item: int):
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id

        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
        """
        if self.local:
            img = Image.open(self.local_root / self.images[item]).convert("RGB")
        else:
            with self.fs.open(f"{self.s3_root}{self.images[item]}") as f:
                img = Image.open(f).convert("RGB")

        label = self.labels[item]

        return img, label

    def __len__(self) -> int:
        return len(self.labels)

    def number_of_classes(self):
        return len(self.class_names)

    def __hash__(self):
        return hash(self.specs_file) + hash(self.s3_root)


def generate_light_easyset(dataset: EasySet, output_dir: Path):
    """
    Generate a light version of an EasySet dataset.
    Args:
        dataset: EasySet dataset
    """
    output_dir.mkdir(exist_ok=False, parents=True)
    for image_id, (image, _) in tqdm(enumerate(dataset)):
        image_name = Path(dataset.images[image_id]).name
        image.save(output_dir / image_name)
