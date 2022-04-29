import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import scipy
import seaborn as sns
import streamlit as st
from torch import nn
from torchvision import transforms

from easyfsl.data_tools import EasySet

st.set_page_config(page_title="Classification Diagnostic", layout="centered")

SPECS_FILE = Path("data/tiered_imagenet/specs/test.json")
IMAGENET_WORDS_PATH = Path("data/tiered_imagenet/specs/words.txt")
DATA_ROOT = Path("data/tiered_imagenet")
METHODS = ["PrototypicalNetworks", "TIM", "PT_MAP", "BDCSPN"]


def plot_task(
    dataset: EasySet,
    testbed_df: pd.DataFrame,
    results: pd.DataFrame,
    task: int,
    class_names,
    only_missclassified,
):
    task_df = testbed_df.loc[lambda df: df.task == task]
    task_queries = task_df.loc[lambda df: ~df.support].image_id.values
    task_results = results.loc[lambda df: df.task_id == task].assign(
        query_id=lambda df: df.image_id.apply(lambda x: task_queries[x])
    )

    st.subheader(f"Support Set")
    support_items = [
        dataset[support_item]
        for support_item in task_df.loc[lambda df: df.support].image_id
    ]
    support_items = {x[1]: x[0] for x in support_items}

    fig, axes = plt.subplots(1, 5)
    for i, (label, image) in enumerate(support_items.items()):
        axes[i].imshow(image)
        axes[i].set_title(
            class_names[label],
            fontsize=8,
        )
        axes[i].axis("off")
    st.pyplot(fig)

    st.subheader(f"Query Classification")
    for label in task_results.true_label.unique():
        with st.expander(label=class_names[label], expanded=False):
            for query in task_results.loc[
                lambda df: df.true_label == label
            ].query_id.unique():
                query_df = task_results.loc[
                    lambda df: df.query_id == query
                ].sort_values("score", ascending=False)
                if (
                    query_df.predicted_label.values[0] != query_df.true_label.values[0]
                    or not only_missclassified
                ):
                    query_image, query_label = dataset[query]
                    assert query_label == query_df.true_label.values[0]
                    scores = scipy.special.softmax(query_df.score.values)

                    fig, axes = plt.subplots(1, 6)
                    axes[0].imshow(query_image)
                    axes[0].axis("off")

                    for i, (_, row) in enumerate(query_df.iterrows()):
                        axes[i + 1].imshow(support_items[row.predicted_label])
                        axes[i + 1].set_title(
                            f"{class_names[int(row.predicted_label)]} \n ({scores[i]:.2f})",
                            fontsize=8,
                        )
                        axes[i + 1].axis("off")
                    st.pyplot(fig)


def get_class_names():
    with open(SPECS_FILE, "r") as file:
        synset_codes = json.load(file)["class_names"]
    words = {}
    with open(IMAGENET_WORDS_PATH, "r") as file:
        for line in file:
            synset, word = line.rstrip().split("\t")
            words[synset] = word.split(",")[0]
    return [words[synset] for synset in synset_codes]


def plot_accuracies_hist(results_df: pd.DataFrame):
    fig, ax = plt.subplots()
    results_df.accuracy.hist(ax=ax, bins=10)
    ax.set_xlabel("accuracy")
    ax.set_ylabel("number of tasks")
    # ax.set_xlim([0, 100])
    st.pyplot(fig)


def compare_methods_on_testbed():
    st.header("Compare Methods on Testbed")
    all_methods_results = pd.concat(
        [
            pd.read_csv(
                DATA_ROOT / "metrics" / method / "task_performances_uniform_1_shot.csv",
                index_col=0,
            ).accuracy.rename(method)
            for method in METHODS
        ],
        axis=1,
    )
    # st.write(all_methods_results.style.format("{:.0%}"))
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
    fails = {
        method: set(
            all_methods_results.loc[
                lambda df: df[method] <= df[method].quantile(threshold)
            ].index.values
        )
        for method in METHODS
    }
    ious = [
        [
            len(fails[method1].intersection(fails[method2])) / len(fails[method1])
            for method2 in METHODS
        ]
        for method1 in METHODS
    ]
    fig, ax = plt.subplots()
    sns.heatmap(
        ious,
        annot=True,
        vmin=0,
        vmax=1,
        cmap="Blues",
        ax=ax,
        xticklabels=METHODS,
        yticklabels=[f"{method}\n({len(fails[method])} total)" for method in METHODS],
    )
    ax.set_title(
        f"Intersection of {threshold:.0%} worst tasks for each method\n ({len(set.intersection(*fails.values()))} shared by all)"
    )
    st.pyplot(fig)

    return all_methods_results


testbed = pd.read_csv(
    DATA_ROOT / "testbeds/testbed_uniform_1_shot.csv", index_col=0
).sort_values(["task", "labels", "support"], ascending=[True, True, False])
dataset = EasySet(specs_file=DATA_ROOT / "specs" / "test.json", training=False)
dataset.transform = transforms.Compose(
    [
        transforms.Resize([84, 84]),
        # transforms.CenterCrop(image_size),
    ]
)

accuracies_for_all_methods = compare_methods_on_testbed().assign(
    average=lambda df: df.mean(axis=1),
    coarsity=pd.read_csv(
        DATA_ROOT / "metrics" / "TIM" / "task_performances_uniform_1_shot.csv",
        index_col=0,
    ).variance,
)

st.subheader("Why are tasks hard?")
st.write(accuracies_for_all_methods)
fig, ax = plt.subplots()
accuracies_for_all_methods.sort_values("average").reset_index(drop=True).assign(
    smooth_coarsity=lambda df: df.coarsity.rolling(window=10).mean(),
    average_in_percents=lambda df: df.average * 100,
).plot.line(ax=ax, y=["average_in_percents", "smooth_coarsity"])
st.pyplot(fig)

st.markdown("""---""")

selected_method = st.sidebar.selectbox("Method", METHODS)
metrics_dir = DATA_ROOT / "metrics" / selected_method

raw_results = pd.read_csv(
    metrics_dir / "raw_results_uniform_1_shot.csv",
    index_col=0,
)
task_results = pd.read_csv(
    metrics_dir / "task_performances_uniform_1_shot.csv",
    index_col=0,
)


class_names = get_class_names()

st.header(f"Task performances for {selected_method}")
col1, col2 = st.columns([2, 3])
with col1:
    st.write(
        task_results.style.format(
            {
                "accuracy": "{:.0%}",
                "variance": "{:.2f}",
            }
        )
    )
with col2:
    plot_accuracies_hist(task_results)

st.header("Dive in a task")
task_choice_columns = st.columns([4, 2])
with task_choice_columns[0]:
    task = st.number_input(
        "Task", key=(0, "Task"), value=0, min_value=0, max_value=testbed.task.max()
    )
with task_choice_columns[1]:
    only_missclassified = st.checkbox(
        "Show only missclassified", key=(0, "Show only missclassified"), value=True
    )
    show_task = st.button("Show task")

if show_task:
    plot_task(dataset, testbed, raw_results, task, class_names, only_missclassified)
