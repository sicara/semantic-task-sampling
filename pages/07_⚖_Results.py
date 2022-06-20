import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns

from st_scripts.st_utils.plot_helpers import render_svg

from st_scripts.st_utils.st_constants import (
    set_slide_page,
    navigation_buttons,
)

set_slide_page()

key = 7

navigation_buttons(key, n_steps=2)

render_svg(f"slides_images/07_{st.session_state.intra_slide_step[key]}.svg")


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
