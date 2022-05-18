import json
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
from torchvision import transforms

from src.easyfsl.data_tools import EasySet, EasySemantics
from st_scripts.st_utils.st_constants import IMAGENET_WORDS_PATH


@st.cache()
def get_class_names(specs_file):
    with open(specs_file, "r") as file:
        synset_codes = json.load(file)["class_names"]
    words = {}
    with open(IMAGENET_WORDS_PATH, "r") as file:
        for line in file:
            synset, word = line.rstrip().split("\t")
            words[synset] = word.split(",")[0]
    return [words[synset] for synset in synset_codes]


@st.cache()
def get_easyset(specs_file, image_size=84):
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


# TODO: caching won't work because the graph gets mutated
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


@st.cache()
def build_coarsity_series(testbed):
    return (
        testbed[["task", "variance", "labels"]]
        .drop_duplicates()
        .groupby("task")
        .variance.mean()
    )


@st.cache()
def build_task_coarsities_df(semantic_testbed, uniform_testbed):
    return pd.DataFrame(
        {
            "with semantic task sampling": build_coarsity_series(semantic_testbed),
            "with uniform task sampling": build_coarsity_series(uniform_testbed),
        }
    )
