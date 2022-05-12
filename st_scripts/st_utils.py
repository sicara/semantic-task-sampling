import json
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.easyfsl.data_tools import EasySet

TESTBEDS_ROOT_DIR = Path("data/tiered_imagenet/testbeds")
SPECS_FILE = Path("data/tiered_imagenet/specs/test.json")
IMAGENET_WORDS_PATH = Path("data/tiered_imagenet/specs/words.txt")


def get_class_names():
    with open(SPECS_FILE, "r") as file:
        synset_codes = json.load(file)["class_names"]
    words = {}
    with open(IMAGENET_WORDS_PATH, "r") as file:
        for line in file:
            synset, word = line.rstrip().split("\t")
            words[synset] = word.split(",")[0]
    return [words[synset] for synset in synset_codes]


def plot_task(dataset: EasySet, testbed_df: pd.DataFrame, task: int, class_names):
    task_df = testbed_df.loc[lambda df: df.task == task]

    support_images = [
        dataset[support_item]
        for support_item in task_df.loc[lambda df: df.support].image_id
    ]

    fig, axes = plt.subplots(1, 5)
    for i, image in enumerate(support_images):
        axes[i].imshow(image[0])
        if class_names:
            axes[i].set_title(
                class_names[image[1]],
                fontsize=8,
            )
        else:
            axes[i].set_title(
                dataset.class_names[image[1]].replace(" ", " \n "),
                fontsize=8,
            )
        axes[i].axis("off")
    st.pyplot(fig)
    st.write(f"Task coarsity: {task_df.variance.mean():.2f}")
