import random

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from pathlib import Path

from networkx.drawing.nx_pydot import graphviz_layout
from pyvis.network import Network
import streamlit.components.v1 as components
from torchvision import transforms

from src.easyfsl.data_tools import EasySet, EasySemantics
from st_scripts.st_utils import (
    TESTBEDS_ROOT_DIR,
    TIERED_TEST_SPECS_FILE,
    get_class_names,
    plot_task,
    IMAGENET_WORDS_PATH,
    MINI_TEST_SPECS_FILE,
)

matplotlib.rcParams["font.family"] = "serif"


@st.cache()
def get_easyset(specs_file, image_size=128):
    dataset = EasySet(specs_file, image_size=image_size)
    dataset.transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
        ]
    )
    return dataset


@st.cache()
def get_testbed(testbed_csv, class_names):
    return pd.read_csv(testbed_csv, index_col=0).assign(
        class_name=lambda df: [class_names[label] for label in df.labels]
    )


# @st.cache()
def get_graph(easy_set: EasySet):
    words = {}
    with open(IMAGENET_WORDS_PATH, "r") as file:
        for line in file:
            synset, word = line.rstrip().split("\t")
            words[synset] = word.split(",")[0]

    return nx.relabel_nodes(
        EasySemantics(
            easy_set, Path("data/tiered_imagenet/specs") / "wordnet.is_a.txt"
        ).dataset_dag,
        words,
    )


st.set_page_config(page_title="Analyze Few-Shot-Learning benchmarks", layout="centered")

st.markdown(  # TODO make this work
    f"""
<style>
    .reportview-container .main .block-container (min-width: 850px){{
        min-width: 1050px;
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.title("By the way, what's in Few-Shot Learning benchmarks?")

st.markdown(
    """
    Since 2018, 98 papers have used miniImageNet as a benchmark. 205 papers have used tieredImageNet. \n
    If you've done any academic research on Few-Shot Image Classification, it is likely that you have used them yourself. 
    You have probably tested some model on hundreds of randomly generated Few-Shot Classification tasks from miniImageNet or tieredImageNet. \n
    But do you know what these tasks look like? \n
    Have you ever wondered what kind of discrimination your model was asked to perform? \n
    If you have not, tis not too late.
    If you have, you're in the right place.
    """
)

tiered_imagenet_class_names = get_class_names(TIERED_TEST_SPECS_FILE)
mini_imagenet_class_names = get_class_names(MINI_TEST_SPECS_FILE)
tiered_dataset = get_easyset(
    TIERED_TEST_SPECS_FILE
)  # TODO: this seems to be runned each time, why?
mini_dataset = get_easyset(MINI_TEST_SPECS_FILE)
uniform_testbed = get_testbed(
    TESTBEDS_ROOT_DIR / "testbed_uniform_1_shot.csv",
    class_names=tiered_imagenet_class_names,
)
mini_testbed = get_testbed(
    "data/mini_imagenet/testbed_uniform_1_shot.csv",
    class_names=tiered_imagenet_class_names,
)
semantic_testbed = get_testbed(
    TESTBEDS_ROOT_DIR / "testbed_1_shot.csv", class_names=tiered_imagenet_class_names
)
tiered_graph = get_graph(tiered_dataset)

st.markdown("---------")

st.header("What do uniformly sampled tasks look like?")

st.markdown(
    """
    Few-Shot Learning benchmarks such as miniImageNet or tieredImageNet evaluate methods on hundreds of Few-Shot Classification tasks. 
    These tasks are sampled uniformly at random from the set of all possible tasks. \n
    This induces a huge bias towards tasks composed of classes that have nothing to do with one another. 
    Classes that you would probably never have to distinguish in any real use case. \n
    See it for yourself. 
    """
)

buttons_cols = st.columns(2)
with buttons_cols[0]:
    if st.button("Draw a task from the test set of tieredImageNet"):
        st.session_state.selected_dataset = "tiered"
        st.session_state.sampled_task = random.randint(0, uniform_testbed.task.max())

with buttons_cols[1]:
    if st.button("Draw a task from the test set of miniImageNet"):
        st.session_state.selected_dataset = "mini"
        st.session_state.sampled_task = random.randint(0, mini_testbed.task.max())

if st.session_state.get("selected_dataset") is not None:
    if st.session_state.selected_dataset == "tiered":
        plot_task(
            tiered_dataset,
            uniform_testbed,
            st.session_state.sampled_task,
            tiered_imagenet_class_names,
        )
    else:
        plot_task(
            mini_dataset,
            mini_testbed,
            st.session_state.sampled_task,
            mini_imagenet_class_names,
        )
    st.markdown(
        """
        If this task looks even remotely like a task you would need to solve ever, please [reach out to me](https://twitter.com/EBennequin).
        
        Because of this shift between those academic benchmark and real life applications of Few-Shot Learning, the performance of a method on those benchmarks is only a distant proxy of its performance on real use cases.
        """
    )

st.markdown("---------")

st.header("Can we do better?")

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

cols = st.columns([2, 3])
with cols[0]:
    st.markdown(
        """
        The classes of tieredImageNet are part of the WordNet graph. \n
        We use this graph to define a semantic distance between classes. \n
        We use this semantic distance to define the coarsity of a task as the mean square distance between the classes constituting the task. \n
        We use this coarsity to sample tasks made of classes that are semantically close to each other. \n
        Play with the coarsity slider. See what kind of tasks we can sample.
        """
    )
with cols[1]:
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
plot_task(tiered_dataset, semantic_testbed, task, tiered_imagenet_class_names)

st.markdown(
    """
    It seems that when you choose a low coarsity, you get a task composed of classes that are semantically close to each other.
    For instance, with the lowest coarsity (8.65), you get the task of discriminating between 5 breeds of dogs.
    On the other hand, when you increase the coarsity, the classes seem to get more distant from one another. \n
    An other way to see this distance is on the WordNet graph. Below you can see the subgraph of WordNet spanned by the classes of tieredImageNet.
    The blue dots are the classes. Highligted in pink, you have the classes that constitute the task you selected.
    Hover any node to see the associated words. \n
    The smaller the coarsity, the closer the classes in the graph.
    """
)

task_classes = semantic_testbed.loc[lambda df: df.task == task].class_name.unique()

fig, ax = plt.subplots()
pos = graphviz_layout(tiered_graph, prog="twopi", root="entity")
pos["physical entity"] = [639.79, 589.48]
pos["entity"] = [600.79, 489.48]

colors = []
sizes = []
for node in tiered_graph:
    tiered_graph.nodes[node]["label"] = " "
    if node == "entity":
        colors.append("black")
        sizes.append(22)
        tiered_graph.nodes[node]["color"] = "black"
        tiered_graph.nodes[node]["size"] = 5
    elif tiered_graph.out_degree(node) == 0:
        if node in task_classes:
            colors.append("#f56cd5")
            tiered_graph.nodes[node]["color"] = "#f56cd5"
            tiered_graph.nodes[node]["shape"] = "diamond"
            # tiered_graph.nodes[node]["label"] = node
        else:
            colors.append("#11aaff")
            tiered_graph.nodes[node]["color"] = "#11aaff"
        sizes.append(14)
    else:
        colors.append("black")
        sizes.append(10)
        tiered_graph.nodes[node]["color"] = "black"
        tiered_graph.nodes[node]["size"] = 5
    tiered_graph.nodes[node]["x"] = pos[node][0]
    tiered_graph.nodes[node]["y"] = pos[node][1]
    tiered_graph.nodes[node]["title"] = node

nt = Network(height="500px", width="670px", bgcolor="#e9f1f7")
nt.from_nx(tiered_graph)
nt.toggle_physics(False)
nt.save_graph("tiered_graph.html")
HtmlFile = open("tiered_graph.html", "r", encoding="utf-8")
source_code = HtmlFile.read()
components.html(source_code, height=550)

st.markdown("---------")
st.header("To go deeper...")

st.markdown(
    """
    This little dashboard was meant to highlight that common Few-Shot Learning benchmarks are strongly biased towards tasks composed of classes that have very distant from each other. \n
    At Sicara, we have seen a wide variety of industrial applications of Few-Shot Learning, but we never encountered a scenario that can be approached by benchmarks presenting this type of bias.
    In fact, in our experience, most applications involve discriminating between classes that are semantically close to each other: plates from plates, tools from tools, carpets from carpets, parts of cars from parts of cars, etc. \n
    There are other benchmarks for fine-grained classifications. And it's OK that some benchmarks contain tasks that are very coarse-grained. 
    But today, tieredImageNet and miniImageNet are wildly used in the literature, and it's important to know what's in there. \n
    
    If you want to know more, check out our paper 
    [Few-Shot Image Classification Benchmarks are Too Far From Reality: Build Back Better with Semantic Task Sampling](https://arxiv.org/abs/2205.05155)
    (presented at the Vision Datasets Understanding Workshop at CVPR 2022).
    """
)
