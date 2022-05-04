import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path

from torchvision import transforms

from src.easyfsl import EasySet
from src.easyfsl.data_tools.danish_fungi import DanishFungi

st.set_page_config(page_title="Analyse testbeds", layout="wide")

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
    st.write(f"Task coarsity: {task_df.variance.mean()}")


def st_explore_testbed(key):
    selected_testbed_path = st.selectbox(
        label="Testbed",
        options=list(TESTBEDS_ROOT_DIR.glob("*.csv")),
        key=(key, "Testbed"),
    )

    class_names = get_class_names()
    image_size = 224
    dataset = EasySet(SPECS_FILE, image_size=image_size)
    dataset.transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            # transforms.CenterCrop(image_size),
        ]
    )

    testbed = pd.read_csv(selected_testbed_path, index_col=0).assign(
        class_name=lambda df: [class_names[label] for label in df.labels]
    )

    testbed_classes = testbed[["task", "variance", "labels"]].drop_duplicates()

    fig, ax = plt.subplots()
    testbed_classes.groupby("task").variance.mean().hist(ax=ax, bins=30)
    ax.set_xlabel("coarsity")
    ax.set_ylabel("number of tasks")
    ax.set_xlim([0, 100])
    st.pyplot(fig)

    fig, ax = plt.subplots()
    testbed_classes.labels.value_counts().hist(ax=ax, bins=10)
    ax.set_xlabel("number of occurrences in the testbed")
    ax.set_ylabel("number of labels")
    ax.set_xlim([110, 200])
    st.pyplot(fig)

    st.write(testbed_classes[["task", "variance"]].drop_duplicates())
    st.write(testbed_classes[["task", "variance"]].drop_duplicates().variance.median())
    task = st.number_input(
        "Task", key=(key, "Task"), value=0, min_value=0, max_value=testbed.task.max()
    )

    plot_task(dataset, testbed, task, class_names)

    return testbed_classes


def st_tiered():
    column_left, column_right = st.columns(2)

    with column_left:
        classes_left = st_explore_testbed(0)
    with column_right:
        class_right = st_explore_testbed(1)

    df = pd.DataFrame({
        "better-tieredImageNet": class_right.labels.value_counts(),
        "uniform sampling": classes_left.labels.value_counts(),
    }) / 50
    st.write(df)
    fig, ax = plt.subplots()
    df.plot.area(ax=ax, stacked=False, color=[ "tomato", "deepskyblue",])
    ax.set_ylim([0, 4])
    ax.set_xlim([0, 159])
    ax.set_xlabel("classes")
    ax.set_ylabel("occurrence (%)")
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    st.pyplot(fig)

def st_fungi():
    column_left, column_right = st.columns(2)

    with column_left:
        selected_testbed_path = Path("data/fungi/testbeds/testbed_uniform_1_shot.csv")

        # class_names = get_class_names()
        image_size = 224
        dataset = DanishFungi()
        dataset.transform = transforms.Compose(
            [
                transforms.Resize([image_size, image_size]),
                # transforms.CenterCrop(image_size),
            ]
        )

        testbed = pd.read_csv(selected_testbed_path, index_col=0).assign(
            class_name=lambda df: [dataset.class_names[label] for label in df.labels]
        )

        testbed_classes = testbed[["task", "variance", "labels"]].drop_duplicates()

        fig, ax = plt.subplots()
        testbed_classes.groupby("task").variance.mean().hist(ax=ax, bins=50)
        ax.set_xlabel("coarsity")
        ax.set_ylabel("number of tasks")
        # ax.set_xlim([0, 100])
        st.pyplot(fig)

        fig, ax = plt.subplots()
        testbed_classes.labels.value_counts().hist(ax=ax, bins=10)
        ax.set_xlabel("number of occurrences in the testbed")
        ax.set_ylabel("number of labels")
        # ax.set_xlim([110, 200])
        st.pyplot(fig)

        st.write(testbed_classes[["task", "variance"]].drop_duplicates())
        st.write(testbed_classes[["task", "variance"]].drop_duplicates().variance.median())
        task = st.number_input(
            "Task", key=(5, "Task"), value=0, min_value=0, max_value=testbed.task.max()
        )

        plot_task(dataset, testbed, task, None)

st_tiered()
