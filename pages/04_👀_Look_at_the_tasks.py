from pathlib import Path

import streamlit as st

from st_scripts.st_utils.data_fetchers import (
    get_testbed,
    get_easyset_expo,
    get_class_names,
)
from st_scripts.st_utils.st_app_blocks import draw_uniform_tasks

from st_scripts.st_utils.st_constants import (
    SICARA_LOGO,
    set_theme,
    TIERED_TEST_SPECS_FILE,
    MINI_TEST_SPECS_FILE,
    TESTBEDS_ROOT_DIR,
    set_slide_page,
    S3_ROOT_TIERED,
    S3_ROOT_MINI,
    vertical_space,
)

set_slide_page()


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


# === MAKE SLIDE ===

vertical_space(20)

st.header("Uniform sampling of classes makes bizarre tasks")

vertical_space(20)
draw_uniform_tasks(
    uniform_testbed,
    mini_testbed,
    tiered_dataset,
    mini_dataset,
    tiered_imagenet_class_names,
    mini_imagenet_class_names,
)
