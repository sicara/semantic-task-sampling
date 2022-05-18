import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.easyfsl.data_tools import EasySet


# TODO: I can't get the caching to work here
def get_task_plot(dataset: EasySet, testbed_df: pd.DataFrame, task: int, class_names):
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
                class_names[image[1]]
                if len(class_names[image[1]]) < 14
                else class_names[image[1]].replace(" ", " \n"),
                fontsize=8,
            )
        else:
            axes[i].set_title(
                dataset.class_names[image[1]].replace(" ", " \n "),
                fontsize=8,
            )
        axes[i].axis("off")

    fig.suptitle(
        f"Task coarsity: {task_df.variance.mean():.2f}",
        verticalalignment="bottom",
        fontsize=7,
        y=0.35,
    )

    st.pyplot(fig)

    return fig


def plot_task(dataset: EasySet, testbed_df: pd.DataFrame, task: int, class_names):
    fig = get_task_plot(dataset, testbed_df, task, class_names)
    # st.pyplot(fig)
