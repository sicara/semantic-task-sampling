import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path

from torchvision import transforms

from src.easyfsl.data_tools.danish_fungi import DanishFungi
from st_scripts.st_utils.plot_helpers import plot_task
from st_scripts.st_utils.st_constants import SICARA_LOGO

st.set_page_config(
    page_title="Compare uniform and semantic testbeds",
    layout="centered",
    page_icon=SICARA_LOGO,
)


def st_fungi():
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


st_fungi()
