import random
from bisect import bisect_left

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

from torchvision import transforms

from src.easyfsl.data_tools import EasySet
from src.easyfsl.data_tools.danish_fungi import DanishFungi
from st_scripts.st_utils import (
    TESTBEDS_ROOT_DIR,
    SPECS_FILE,
    get_class_names,
    plot_task,
)

# pd.options.plotting.backend = "plotly"

st.set_page_config(page_title="Analyze Few-Shot-Learning benchmarks", layout="wide")

st.title("What do we talk about when we talk about tieredImageNet?")

st.markdown(
    "Since 2018, 98 papers have used miniImageNet as a benchmark. 205 papers have used tieredImageNet. "
    "If you've done any research on Few-Shot Image Classification, it is likely that you have used them yourself. "
    "You have probably tested some model on hundreds of randomly generated Few-Shot Classification tasks from miniImageNet or tieredImageNet. "
    "But do you know what these tasks look like? "
    "Have you ever wondered what kind of discrimination your model was asked to perform? "
    "If you have not, tis not too late. "
    "If you have, you're in the right place. "
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

st.header("What do uniformly sampled tasks look like?")

st.markdown(
    "Few-Shot Learning benchmarks such as miniImageNet or tieredImageNet evaluate methods on hundreds of Few-Shot Classification tasks. "
    "These tasks are sampled uniformly at random from the set of all possible tasks. "
    "This induces a huge bias towards tasks composed of classes that have nothing to do with one another. "
    "Classes that you would probably never have to distinguish in any real use case. "
    "See it for yourself. "
)

uniform_testbed = pd.read_csv(
    TESTBEDS_ROOT_DIR / "testbed_uniform_1_shot.csv", index_col=0
).assign(class_name=lambda df: [class_names[label] for label in df.labels])

if st.button("Draw a task from tieredImageNet's test set"):
    plot_task(
        dataset,
        uniform_testbed,
        random.randint(0, uniform_testbed.task.max()),
        class_names,
    )
    st.markdown(
        "If this task looks even remotely like a task you would need to solve ever, please [reach out to me](https://twitter.com/EBennequin). "
        "Because of this shift between those academic benchmark and real life applications of Few-Shot Learning, the performance of a method on those benchmarks is only a distant proxy of its performance on real use cases. "
    )
# if st.button("Draw a task from miniImageNet's test set"):
#     1 + 1

st.markdown("---------")

st.header("Can we do better?")
st.markdown(
    "The classes of tieredImageNet are part of the WordNet graph.  "
    "We use this graph to define a semantic distance between classes.  "
    "We use this semantic distance to define the coarsity of a task as the mean square distance between the classes constituting the task.  "
    "We use this coarsity to sample tasks made of classes that are semantically close to each other.  "
    "Play with the coarsity slider. See what kind of tasks we can sample.  "
)

semantic_testbed = pd.read_csv(
    TESTBEDS_ROOT_DIR / "testbed_1_shot.csv", index_col=0
).assign(class_name=lambda df: [class_names[label] for label in df.labels])

# testbed_classes = semantic_testbed[["task", "variance", "labels"]].drop_duplicates()

# fig, ax = plt.subplots()
# fig = px.histogram(testbed_classes.groupby("task").variance.mean(), labels=
# {"value": "coarsity", "count": "number of tasks"})
# # fig = testbed_classes.groupby("task").variance.mean().plot(
# #     kind="hist",
# #     bins=30,
# #     xlabel="coarsity",
# #     ylabel="number of tasks",
# #     xlim=[0, 100],
# # )
# # fig.set_xlabel("coarsity")
# # fig.set_ylabel("number of tasks")
# # fig.set_xlim([0, 100])
# st.plotly_chart(fig)

task_coarsities = pd.DataFrame(
    {
        "with semantic task sampling": (
            semantic_testbed[["task", "variance", "labels"]]
            .drop_duplicates()
            .groupby("task")
            .variance.mean()
        ),
        "with uniform task sampling": (
            uniform_testbed[["task", "variance", "labels"]]
            .drop_duplicates()
            .groupby("task")
            .variance.mean()
        ),
    }
)

cols = st.columns(2)
with cols[0]:
    fig, ax = plt.subplots()
    task_coarsities.plot.hist(
        ax=ax,
        bins=30,
        alpha=0.8,
        color=[
            "#f56cd5",
            "#11aaff",
        ],
    )
    ax.set_xlabel("coarsity")
    ax.set_ylabel("number of tasks")
    ax.set_xlim([0, 100])
    st.pyplot(fig)
with cols[1]:
    step = 0.1
    semantic_task_coarsities = task_coarsities["with semantic task sampling"]

    selected_coarsity = st.slider(
        "Coarsity",
        min_value=float(semantic_task_coarsities.min()),
        max_value=float(semantic_task_coarsities.max()),
        value=float(semantic_task_coarsities.median()),
        step=step,
    )

    sorted_task_coarsity = semantic_task_coarsities.sort_values()
    index_in_sorted_series = sorted_task_coarsity.searchsorted(selected_coarsity)
    task = random.sample(
        set(
            sorted_task_coarsity.loc[
                (
                    sorted_task_coarsity
                    >= sorted_task_coarsity.iloc[index_in_sorted_series] - step
                )
                & (
                    sorted_task_coarsity
                    <= sorted_task_coarsity.iloc[index_in_sorted_series] + step
                )
            ].index
        ),
        k=1,
    )[0]
    plot_task(dataset, semantic_testbed, task, class_names)
