import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pathlib import Path

from torchvision import transforms

from src.easyfsl.data_tools.danish_fungi import DanishFungi
from st_scripts.st_utils.data_fetchers import get_testbed
from st_scripts.st_utils.plot_helpers import (
    plot_coarsities_hist,
    plot_occurrences_hist,
    plot_wide_task,
)
from st_scripts.st_utils.st_constants import SICARA_LOGO, FUNGI_1_SHOT_TESTBED

title = "Fungi testbed"
st.set_page_config(
    page_title=title,
    layout="centered",
    page_icon=SICARA_LOGO,
)

st.title(title)


def st_fungi():

    image_size = 224
    dataset = DanishFungi()
    dataset.transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
        ]
    )

    testbed = get_testbed(FUNGI_1_SHOT_TESTBED, dataset.class_names)

    testbed_classes = testbed[["task", "variance", "labels"]].drop_duplicates()
    task_coarsities = (
        testbed_classes[["task", "variance"]]
        .drop_duplicates()
        .set_index("task")
        .rename(columns={"variance": "coarsity"})
    )

    plot_coarsities_hist(task_coarsities.coarsity)

    plot_occurrences_hist(testbed_classes)

    st.subheader(f"Coarsity by task (median: {task_coarsities.coarsity.median():.2f})")
    st.write(task_coarsities.style.format("{:.2f}"))

    task = st.number_input(
        "Task", key=(5, "Task"), value=0, min_value=0, max_value=testbed.task.max()
    )

    plot_wide_task(dataset, testbed, task, dataset.class_names, 100)


st_fungi()
