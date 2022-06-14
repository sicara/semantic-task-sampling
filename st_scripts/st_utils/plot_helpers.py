import base64
from pathlib import Path
from typing import Union

import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt

from src.easyfsl.data_tools import EasySet
from st_scripts.st_utils.st_constants import PRIMARY_APP_COLOR, SECONDARY_APP_COLOR


def build_subplots_grid(n_way: int, max_columns: int = 5):
    return plt.subplots(
        n_way // max_columns, max_columns, figsize=(5, 2 * n_way // max_columns)
    )


# @st.cache
def get_support_images(dataset, task_df):
    return [
        dataset[support_item]
        for support_item in task_df.loc[lambda df: df.support].image_id
    ]


def plot_task(
    dataset: EasySet,
    testbed_df: pd.DataFrame,
    task: int,
    class_names,
    display_coarsity=True,
):
    task_df = testbed_df.loc[lambda df: df.task == task]

    support_images = get_support_images(dataset, task_df)

    fig, axes = plt.subplots(1, 5)
    for i, image in enumerate(support_images):
        axes[i].imshow(image[0])
        if class_names:
            axes[i].set_title(
                class_names[image[1]]
                if len(class_names[image[1]]) < 14
                else class_names[image[1]].replace(" ", " \n"),
                fontsize=8,
            )
        axes[i].axis("off")

    if display_coarsity:
        fig.suptitle(
            f"Task coarsity: {task_df.variance.mean():.2f}",
            verticalalignment="bottom",
            fontsize=7,
            y=0.35,
        )

    st.pyplot(fig)

    return fig


def plot_wide_task(
    dataset: EasySet,
    testbed_df: pd.DataFrame,
    task: int,
    class_names,
    n_way: int = 5,
    max_columns: int = 5,
):
    task_df = testbed_df.loc[lambda df: df.task == task]

    support_images = [
        dataset[support_item]
        for support_item in task_df.loc[lambda df: df.support].image_id
    ]

    fig, axes = plt.subplots(
        n_way // max_columns, max_columns, figsize=(5, 2 * n_way // max_columns)
    )

    for i, image in enumerate(support_images):
        row = i // max_columns
        column = i % max_columns
        axes[row][column].imshow(image[0])
        if class_names:
            axes[row][column].set_title(
                class_names[image[1]]
                if len(class_names[image[1]]) < 14
                else class_names[image[1]].replace(" ", " \n"),
                fontsize=6,
            )
        axes[row][column].axis("off")

    st.write(
        f"Task coarsity: {task_df.variance.mean():.2f}",
    )

    st.pyplot(fig)

    return fig


def plot_coarsities_hist(task_coarsities, xlim=None):
    fig, ax = plt.subplots()
    sns.histplot(
        task_coarsities,
        ax=ax,
        kde=True,
        linewidth=0,
        color=SECONDARY_APP_COLOR,
        binwidth=2,
    )
    ax.set_xlabel("coarsity")
    ax.set_ylabel("number of tasks")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim((0, 800))
    st.pyplot(fig)


def plot_occurrences_hist(testbed_classes):
    fig, ax = plt.subplots()
    sns.histplot(
        testbed_classes.labels.value_counts(),
        ax=ax,
        bins=10,
        kde=True,
        linewidth=0,
    )
    ax.set_xlabel("number of occurrences in the testbed")
    ax.set_ylabel("number of labels")
    st.pyplot(fig)


def plot_occurrences_comparison(occurrences_df):
    fig, ax = plt.subplots()
    occurrences_df.plot.area(
        ax=ax,
        stacked=False,
        color=[
            "tomato",
            "deepskyblue",
        ],
    )
    ax.set_ylim([0, 4])
    ax.set_xlim([0, 159])
    ax.set_xlabel("classes")
    ax.set_ylabel("occurrence (%)")
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    st.pyplot(fig)


def render_svg(src: Union[Path, str]):

    with open(src, "r") as f:
        lines = f.readlines()
        svg = "".join(lines)
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    st.write(html, unsafe_allow_html=True)
