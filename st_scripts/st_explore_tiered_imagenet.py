import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from torchvision import transforms

from src.easyfsl.data_tools import EasySet
from st_scripts.st_utils.st_constants import (
    TESTBEDS_ROOT_DIR,
    TIERED_TEST_SPECS_FILE,
    SICARA_LOGO,
)
from st_scripts.st_utils.plot_helpers import (
    plot_task,
    plot_coarsities_hist,
    plot_occurrences_comparison,
)
from st_scripts.st_utils.data_fetchers import get_class_names, get_testbed

title = "Compare uniform and semantic 1-shot testbeds"
st.set_page_config(
    page_title=title,
    layout="wide",
    page_icon=SICARA_LOGO,
)
st.title(title)


def st_explore_testbed(testbed_path):
    class_names = get_class_names(TIERED_TEST_SPECS_FILE)
    image_size = 224
    dataset = EasySet(TIERED_TEST_SPECS_FILE, image_size=image_size)
    dataset.transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
            # transforms.CenterCrop(image_size),
        ]
    )

    testbed = get_testbed(testbed_path, class_names)

    testbed_classes = testbed[["task", "variance", "labels"]].drop_duplicates()

    task_coarsities = (
        testbed_classes[["task", "variance"]]
        .drop_duplicates()
        .set_index("task")
        .rename(columns={"variance": "coarsity"})
    )

    plot_coarsities_hist(task_coarsities.coarsity, [0, 100])

    st.subheader(f"Coarsity by task (median: {task_coarsities.coarsity.median():.2f})")
    st.write(task_coarsities.style.format("{:.2f}"))

    st.subheader("Look at a task's support set")
    task = st.number_input(
        "Task",
        key=(testbed_path.name, "Task"),
        value=0,
        min_value=0,
        max_value=testbed.task.max(),
    )

    plot_task(dataset, testbed, task, class_names)

    return testbed_classes


def st_tiered():
    column_left, column_right = st.columns(2)

    with column_left:
        st.header("Testbed with uniform task sampling")
        classes_left = st_explore_testbed(
            TESTBEDS_ROOT_DIR / "testbed_uniform_1_shot.csv"
        )
    with column_right:
        st.header("Testbed with semantic task sampling")
        class_right = st_explore_testbed(TESTBEDS_ROOT_DIR / "testbed_1_shot.csv")

    st.header("Compare the class balance of the testbeds")
    plot_occurrences_comparison(
        pd.DataFrame(
            {
                "semantic sampling": class_right.labels.value_counts(),
                "uniform sampling": classes_left.labels.value_counts(),
            }
        )
        / 50
    )


st_tiered()
