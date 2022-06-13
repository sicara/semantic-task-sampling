import streamlit as st
from st_scripts.st_utils.data_fetchers import (
    get_testbed,
    get_easyset_expo,
    get_class_names,
    build_task_coarsities_df,
)
from st_scripts.st_utils.plot_helpers import plot_coarsities_hist, plot_task
from st_scripts.st_utils.st_app_blocks import (
    draw_semantic_task,
)

from st_scripts.st_utils.st_constants import (
    TIERED_TEST_SPECS_FILE,
    TESTBEDS_ROOT_DIR,
    set_slide_page,
    SEMANTIC_SLIDER_STEP,
    S3_ROOT_TIERED,
    navigation_buttons,
)
from st_scripts.st_utils.st_wordings import st_divider

set_slide_page()

key = 6

# === FETCH ALL THE DATA WE NEED ===

tiered_imagenet_class_names = get_class_names(TIERED_TEST_SPECS_FILE)
tiered_dataset = get_easyset_expo(TIERED_TEST_SPECS_FILE, S3_ROOT_TIERED)

uniform_testbed = get_testbed(
    TESTBEDS_ROOT_DIR / "testbed_uniform_1_shot_expo.csv",
    class_names=tiered_imagenet_class_names,
)
semantic_testbed = get_testbed(
    TESTBEDS_ROOT_DIR / "testbed_1_shot_expo.csv",
    class_names=tiered_imagenet_class_names,
)

task_coarsities = build_task_coarsities_df(semantic_testbed, uniform_testbed)

# === NAVIGATION BUTTONS ===

navigation_buttons(key, n_steps=1)

# === ACTION ===
col1, col2 = st.columns([2, 2])
with col1:
    if st.session_state.intra_slide_step[key] == 0:
        plot_coarsities_hist(
            task_coarsities["with uniform task sampling"], xlim=(5, 100)
        )
    else:
        plot_coarsities_hist(task_coarsities, xlim=(5, 100))

if st.session_state.intra_slide_step[key] > 0:
    with col2:

        st.subheader("Semantic Task Sampling")
        st.markdown(
            r"""
        Classes $$i$$ and $$j$$ are co-sampled in a task with a probability proportional to their potential:
        """
        )
        st.latex(
            r"""
        \mathcal{P}_0(i,j) = e^{-\alpha D^{JC}(c_i, c_j)}
        """
        )
        st.markdown(
            r"""
        with $$\alpha \in \mathbb R_+$$ an arbitrary scalar.
        """
        )

        st_divider()

        default_coarsity = float(
            task_coarsities["with semantic task sampling"].median()
        )

        selected_coarsity = st.slider(
            "Sample a task",
            min_value=float(task_coarsities["with semantic task sampling"].min()),
            max_value=float(task_coarsities["with semantic task sampling"].max()),
            value=default_coarsity,
            step=SEMANTIC_SLIDER_STEP,
        )

    if selected_coarsity != default_coarsity:
        task = draw_semantic_task(
            task_coarsities["with semantic task sampling"], selected_coarsity
        )

        plot_task(tiered_dataset, semantic_testbed, task, tiered_imagenet_class_names)
