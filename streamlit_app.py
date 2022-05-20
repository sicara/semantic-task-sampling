import streamlit as st

from st_scripts.st_utils.st_constants import (
    TESTBEDS_ROOT_DIR,
    TIERED_TEST_SPECS_FILE,
    MINI_TEST_SPECS_FILE,
    S3_ROOT_MINI,
    S3_ROOT_TIERED,
    SICARA_LOGO,
    set_theme,
)
from st_scripts.st_utils.st_app_blocks import (
    draw_uniform_tasks,
    show_semantic_tasks,
    plot_semantic_graph,
)
from st_scripts.st_utils.plot_helpers import plot_coarsities_hist
from st_scripts.st_utils.data_fetchers import (
    get_class_names,
    get_testbed,
    build_task_coarsities_df,
    get_easyset_expo,
)
from st_scripts.st_utils.st_wordings import WORDINGS, st_divider

# === APP SETUP ===

st.set_page_config(
    page_title="Analyze Few-Shot-Learning benchmarks",
    layout="centered",
    page_icon=SICARA_LOGO,
)

set_theme()

# === FETCH ALL THE DATA WE NEED ===

tiered_imagenet_class_names = get_class_names(TIERED_TEST_SPECS_FILE)
mini_imagenet_class_names = get_class_names(MINI_TEST_SPECS_FILE)
tiered_dataset = get_easyset_expo(TIERED_TEST_SPECS_FILE, S3_ROOT_TIERED)
mini_dataset = get_easyset_expo(MINI_TEST_SPECS_FILE, S3_ROOT_MINI)

uniform_testbed = get_testbed(
    TESTBEDS_ROOT_DIR / "testbed_uniform_1_shot_expo.csv",
    class_names=tiered_imagenet_class_names,
)
mini_testbed = get_testbed(
    "data/mini_imagenet/testbed_uniform_1_shot.csv",
    class_names=tiered_imagenet_class_names,
)
semantic_testbed = get_testbed(
    TESTBEDS_ROOT_DIR / "testbed_1_shot_expo.csv",
    class_names=tiered_imagenet_class_names,
)

task_coarsities = build_task_coarsities_df(semantic_testbed, uniform_testbed)


# === AND NOW: THE APP ===

st.title("By the way, what's in Few-Shot Learning benchmarks?")

st.markdown(WORDINGS["app_intro"])

st_divider()
st.header("Uniformly sampled tasks do not reflect real-world use cases")

st.markdown(WORDINGS["uniform_tasks"])

draw_uniform_tasks(
    uniform_testbed,
    mini_testbed,
    tiered_dataset,
    mini_dataset,
    tiered_imagenet_class_names,
    mini_imagenet_class_names,
)

st_divider()
st.header("Can we do better?")

cols = st.columns([2, 3])
with cols[0]:
    st.markdown(WORDINGS["explain_semantic_sampling"])

with cols[1]:
    plot_coarsities_hist(task_coarsities)

st.markdown(WORDINGS["introduce_slider"])

task = show_semantic_tasks(
    semantic_task_coarsities=task_coarsities["with semantic task sampling"],
    dataset=tiered_dataset,
    testbed=semantic_testbed,
    class_names=tiered_imagenet_class_names,
)

st.markdown(WORDINGS["semantic_graph"])

plot_semantic_graph(task, semantic_testbed)

st_divider()
st.header("To go deeper...")

st.markdown(WORDINGS["app_conclusion"])

st_divider()
st.subheader("About me")
about_cols = st.columns([9, 1])

with about_cols[1]:
    st.image("https://ebennequin.github.io/images/profile.jpeg")
    st.image(SICARA_LOGO)
with about_cols[0]:
    st.markdown(WORDINGS["about"])
