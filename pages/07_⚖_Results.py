import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns

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
    PRIMARY_APP_COLOR,
)
from st_scripts.st_utils.st_wordings import st_divider

set_slide_page()

key = 7

navigation_buttons(key, n_steps=1)

st.image(
    f"slides_images/07_{st.session_state.intra_slide_step[key]}.png",
    use_column_width="always",
)


# === ACTION ===


def plot_results_as_heatmap(results_csv="slides_images/results.csv"):
    """
    How we got the plot used in this slide.
    """
    results = pd.read_csv(results_csv, index_col=0).rename(
        index={"Transductive Finetuning": "Transductive\nFinetuning"}
    )

    fig, ax = plt.subplots()
    sns.heatmap(
        results,
        annot=True,
        fmt=".2f",
        annot_kws={"color": "white"},
        vmin=20,
        vmax=75,
        cmap="Greens",
        ax=ax,
        cbar=False,
        xticklabels=False,
    )
    st.pyplot(fig)
