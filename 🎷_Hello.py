from pathlib import Path

import streamlit as st

from st_scripts.st_utils.st_constants import (
    TESTBEDS_ROOT_DIR,
    TIERED_TEST_SPECS_FILE,
    MINI_TEST_SPECS_FILE,
    S3_ROOT_MINI,
    S3_ROOT_TIERED,
    SICARA_LOGO,
    set_theme,
    set_slide_page,
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

set_slide_page()
