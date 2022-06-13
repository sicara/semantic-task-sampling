import json
import random

import streamlit as st
from networkx.readwrite import json_graph
from pyvis.network import Network
from streamlit.components import v1 as components

from st_scripts.st_utils.plot_helpers import plot_task
from st_scripts.st_utils.st_constants import (
    PRIMARY_APP_COLOR,
    SECONDARY_APP_COLOR,
    SEMANTIC_SLIDER_STEP,
    TIERED_GRAPH_PATH,
)
from st_scripts.st_utils.st_wordings import WORDINGS


def draw_uniform_tasks(
    tiered_uniform_testbed,
    mini_uniform_testbed,
    tiered_dataset,
    mini_dataset,
    tiered_class_names,
    mini_class_names,
):
    buttons_cols = st.columns(2)
    with buttons_cols[0]:
        if st.button("tieredImageNet"):
            st.session_state.selected_dataset = "tiered"
            st.session_state.sampled_task = random.randint(
                0, tiered_uniform_testbed.task.max()
            )
    with buttons_cols[1]:
        if st.button("miniImageNet"):
            st.session_state.selected_dataset = "mini"
            st.session_state.sampled_task = random.randint(
                0, mini_uniform_testbed.task.max()
            )
    if st.session_state.get("selected_dataset") is not None:
        if st.session_state.selected_dataset == "tiered":
            plot_task(
                tiered_dataset,
                tiered_uniform_testbed,
                st.session_state.sampled_task,
                tiered_class_names,
                display_coarsity=False,
            )
        else:
            plot_task(
                mini_dataset,
                mini_uniform_testbed,
                st.session_state.sampled_task,
                mini_class_names,
                display_coarsity=False,
            )

        st.write("These 5 classes are probably not linked in any way.")


@st.experimental_memo
def draw_semantic_task(task_coarsities, input_coarsity):
    sorted_task_coarsity = task_coarsities.sort_values()
    index_in_sorted_series = sorted_task_coarsity.searchsorted(input_coarsity)
    return random.sample(
        set(
            sorted_task_coarsity.loc[
                (
                    sorted_task_coarsity
                    >= sorted_task_coarsity.iloc[index_in_sorted_series]
                    - SEMANTIC_SLIDER_STEP
                )
                & (
                    sorted_task_coarsity
                    <= sorted_task_coarsity.iloc[index_in_sorted_series]
                    + SEMANTIC_SLIDER_STEP
                )
            ].index
        ),
        k=1,
    )[0]


def show_semantic_tasks(semantic_task_coarsities, dataset, testbed, class_names):
    selected_coarsity = st.slider(
        "Coarsity",
        min_value=float(semantic_task_coarsities.min()),
        max_value=float(semantic_task_coarsities.max()),
        value=float(semantic_task_coarsities.median()),
        step=SEMANTIC_SLIDER_STEP,
    )
    task = draw_semantic_task(semantic_task_coarsities, selected_coarsity)

    plot_task(dataset, testbed, task, class_names)

    return task


def plot_semantic_graph():

    with open(TIERED_GRAPH_PATH, "r") as f:
        graph = json_graph.node_link_graph(json.load(f))
    colors = []
    sizes = []
    for node in graph:
        graph.nodes[node]["label"] = " "
        if node == "entity":
            colors.append("black")
            sizes.append(22)
            graph.nodes[node]["color"] = SECONDARY_APP_COLOR
            graph.nodes[node]["size"] = 5
        elif graph.out_degree(node) == 0:
            # if node in task_classes:
            colors.append(PRIMARY_APP_COLOR)
            graph.nodes[node]["color"] = PRIMARY_APP_COLOR
            # graph.nodes[node]["shape"] = "diamond"
            # else:
            #     colors.append(SECONDARY_APP_COLOR)
            #     graph.nodes[node]["color"] = SECONDARY_APP_COLOR
            sizes.append(14)
        else:
            colors.append("black")
            sizes.append(10)
            graph.nodes[node]["color"] = SECONDARY_APP_COLOR
            graph.nodes[node]["size"] = 5

    nt = Network(height="500px", width="100%", bgcolor="white")
    nt.from_nx(graph)
    nt.toggle_physics(False)
    nt.save_graph("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=550)
