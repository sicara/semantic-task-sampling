import random

import streamlit as st
from networkx.drawing.nx_pydot import graphviz_layout
from pyvis.network import Network
from streamlit.components import v1 as components

from st_scripts.st_utils.data_fetchers import get_graph
from st_scripts.st_utils.plot_helpers import plot_task
from st_scripts.st_utils.st_constants import PRIMARY_APP_COLOR, SECONDARY_APP_COLOR
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
        if st.button("Draw a task from the test set of tieredImageNet"):
            st.session_state.selected_dataset = "tiered"
            st.session_state.sampled_task = random.randint(
                0, tiered_uniform_testbed.task.max()
            )
    with buttons_cols[1]:
        if st.button("Draw a task from the test set of miniImageNet"):
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
            )
        else:
            plot_task(
                mini_dataset,
                mini_uniform_testbed,
                st.session_state.sampled_task,
                mini_class_names,
            )
        st.markdown(WORDINGS["after_uniform_task"])


def show_semantic_tasks(semantic_task_coarsities, dataset, testbed, class_names):
    step = 0.1
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
    plot_task(dataset, testbed, task, class_names)

    return task


def plot_semantic_graph(task, testbed, dataset):
    task_classes = testbed.loc[lambda df: df.task == task].class_name.unique()

    graph = get_graph(dataset)

    pos = graphviz_layout(graph, prog="twopi", root="entity")
    pos["physical entity"] = [639.79, 589.48]
    pos["entity"] = [600.79, 489.48]
    colors = []
    sizes = []
    for node in graph:
        graph.nodes[node]["label"] = " "
        if node == "entity":
            colors.append("black")
            sizes.append(22)
            graph.nodes[node]["color"] = "black"
            graph.nodes[node]["size"] = 5
        elif graph.out_degree(node) == 0:
            if node in task_classes:
                colors.append(PRIMARY_APP_COLOR)
                graph.nodes[node]["color"] = PRIMARY_APP_COLOR
                graph.nodes[node]["shape"] = "diamond"
                # graph.nodes[node]["label"] = node
            else:
                colors.append(SECONDARY_APP_COLOR)
                graph.nodes[node]["color"] = SECONDARY_APP_COLOR
            sizes.append(14)
        else:
            colors.append("black")
            sizes.append(10)
            graph.nodes[node]["color"] = "black"
            graph.nodes[node]["size"] = 5
        graph.nodes[node]["x"] = pos[node][0]
        graph.nodes[node]["y"] = pos[node][1]
        graph.nodes[node]["title"] = node
    nt = Network(height="500px", width="100%", bgcolor="#e9f1f7")
    nt.from_nx(graph)
    nt.toggle_physics(False)
    nt.save_graph("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")
    source_code = HtmlFile.read()
    components.html(source_code, height=550)
