import json
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
from networkx.drawing.nx_pydot import graphviz_layout
from torchvision import transforms

from src.easyfsl.data_tools import EasySet, EasySemantics
from src.easyfsl.data_tools.easy_set_light import EasySetExpo
from st_scripts.st_utils.st_constants import IMAGENET_WORDS_PATH


@st.experimental_memo
def get_class_names(specs_file):
    with open(specs_file, "r") as file:
        synset_codes = json.load(file)["class_names"]
    words = {}
    with open(IMAGENET_WORDS_PATH, "r") as file:
        for line in file:
            synset, word = line.rstrip().split("\t")
            words[synset] = word.split(",")[0]
    return [words[synset] for synset in synset_codes]


@st.experimental_memo
def get_easyset(specs_file, image_size=84):
    dataset = EasySet(specs_file, image_size=image_size)
    dataset.transform = transforms.Compose(
        [
            transforms.Resize([image_size, image_size]),
        ]
    )
    return dataset


@st.experimental_memo
def get_testbed(testbed_csv, class_names):
    return pd.read_csv(testbed_csv, index_col=0).assign(
        class_name=lambda df: [class_names[label] for label in df.labels]
    )


def get_graph(easy_set: EasySet):
    """
    This is how we got the graph in data/tiered_imagenet/specs/semantic_graph.json
    """
    words = {}
    with open(IMAGENET_WORDS_PATH, "r") as file:
        for line in file:
            synset, word = line.rstrip().split("\t")
            words[synset] = word.split(",")[0]

    graph = nx.relabel_nodes(
        EasySemantics(
            easy_set, Path("data/tiered_imagenet/specs") / "wordnet.is_a.txt"
        ).dataset_dag,
        words,
    )
    pos = graphviz_layout(graph, prog="twopi", root="entity")
    pos["physical entity"] = [639.79, 589.48]
    pos["entity"] = [600.79, 489.48]

    for node in graph:
        graph.nodes[node]["x"] = pos[node][0]
        graph.nodes[node]["y"] = pos[node][1]
        graph.nodes[node]["title"] = node

    return graph


@st.experimental_memo
def build_coarsity_series(testbed):
    return (
        testbed[["task", "variance", "labels"]]
        .drop_duplicates()
        .groupby("task")
        .variance.mean()
    )


@st.experimental_memo
def build_task_coarsities_df(semantic_testbed, uniform_testbed):
    return pd.DataFrame(
        {
            "with semantic task sampling": build_coarsity_series(semantic_testbed),
            "with uniform task sampling": build_coarsity_series(uniform_testbed),
        }
    )


@st.experimental_memo
def get_easyset_expo(*args, **kwargs):
    return EasySetExpo(*args, **kwargs)
