import pickle
from pathlib import Path
from typing import Union

import s3fs
from PIL import Image
from tqdm import tqdm

from src.easyfsl.data_tools import EasySet


class EasySetExpo(EasySet):
    """
    EasySet but we fetch the data from S3 and we don't do any preprocessing of images.
    """

    def __init__(self, specs_file: Union[Path, str], s3_root: str):
        """
        Args:
            specs_file: path to the JSON file needed by EasySet
            s3_root: s3://bucket/prefix/
        """
        super().__init__(specs_file)

        self.fs = s3fs.S3FileSystem(anon=False)
        self.s3_root = s3_root

    def __getitem__(self, item: int):
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id

        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
        """
        with self.fs.open(f"{self.s3_root}{Path(self.images[item]).name}") as f:
            img = Image.open(f).convert("RGB")

        label = self.labels[item]

        return img, label

    def __len__(self) -> int:
        return len(self.labels)

    def number_of_classes(self):
        return len(self.class_names)


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
