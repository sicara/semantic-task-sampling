from pathlib import Path

import matplotlib
import seaborn as sns
import streamlit as st

# ==== CONSTANTS ====

# - Paths -

TESTBEDS_ROOT_DIR = Path("data/tiered_imagenet/testbeds")
FUNGI_1_SHOT_TESTBED = Path("data/fungi/testbeds/testbed_uniform_1_shot.csv")

TIERED_TEST_SPECS_FILE = Path("data/tiered_imagenet/specs/test.json")
MINI_TEST_SPECS_FILE = Path("data/mini_imagenet/test.json")
IMAGENET_WORDS_PATH = Path("data/tiered_imagenet/specs/words.txt")
TIERED_GRAPH_PATH = Path("data/tiered_imagenet/specs/semantic_graph.json")

S3_ROOT_MINI = "s3://thesis-etienne/mini_light/"
S3_ROOT_TIERED = "s3://thesis-etienne/tiered_light/"

# - Theme -

PRIMARY_APP_COLOR = "#f56cd5"
SECONDARY_APP_COLOR = "#11aaff"
SICARA_LOGO = "https://theodo.github.io/signature/images/logoSicara.png"

SEMANTIC_SLIDER_STEP = 0.1


def set_seaborn_theme():
    sns.set(
        style="ticks",
        palette=sns.color_palette(
            [
                PRIMARY_APP_COLOR,
                SECONDARY_APP_COLOR,
            ]
        ),
        font="serif",
    )
    sns.despine()


def set_theme():
    # Default parameters for plotting libraries
    matplotlib.rcParams["font.family"] = "serif"
    set_seaborn_theme()

    # Custom app CSS
    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container {{
            width: 1050px;
            max-width: 90%;
        }}
        
        .stButton {{
            display: flex;
            justify-content: space-around;
        }}
        
        .css-2tp4zm a {{
            color: {SECONDARY_APP_COLOR};
        }}
        
        .css-2tp4zm a:hover {{
            color: {PRIMARY_APP_COLOR};
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )
