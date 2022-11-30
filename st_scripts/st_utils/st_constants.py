from pathlib import Path

import matplotlib
import seaborn as sns
import streamlit as st

# ==== CONSTANTS ====

# - Paths -
from streamlit.components import v1 as components

TESTBEDS_ROOT_DIR = Path("data/tiered_imagenet/testbeds")
FUNGI_1_SHOT_TESTBED = Path("data/fungi/testbeds/testbed_uniform_1_shot.csv")

TIERED_TEST_SPECS_FILE = Path("data/tiered_imagenet/specs/test.json")
MINI_TEST_SPECS_FILE = Path("data/mini_imagenet/test.json")
IMAGENET_WORDS_PATH = Path("data/tiered_imagenet/specs/words.txt")
TIERED_GRAPH_PATH = Path("data/tiered_imagenet/specs/semantic_graph.json")

S3_ROOT_MINI = "s3://thesis-etienne/mini_light/"
S3_ROOT_TIERED = "s3://thesis-etienne/tiered_light/"

LOCAL_ROOT_MINI = Path("data/mini_imagenet/light")
LOCAL_ROOT_TIERED = Path("data/tiered_imagenet/light")

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
        # .css-168ft4l {{
        #     width: 25rem;
        # }}
        .main .block-container {{
            padding-top: 30px;
        }}
        .block-container .stButton {{
            font-size: 10pt;
            # position: fixed;
            # bottom: 10px;
        }}
        .main .stButton {{
            display: flex;
            justify-content: space-around;
            font-size: 35pt;
        }}
        .css-1h9o3pk {{
            color: {SECONDARY_APP_COLOR};
            border-color: {SECONDARY_APP_COLOR};
        }}
        .css-19effar a {{
            color: {SECONDARY_APP_COLOR};
        }}
        a:hover {{
            color: {PRIMARY_APP_COLOR};
        }}
        .block-container .stVerticalBlock p {{
            text-align: justify;
        }}
        .main p {{
            font-size: 20pt;
        }}
        .streamlit-expanderHeader {{
            font-weight: bold;
            font-size: 20pt;
        }}
        .math-display {{
            font-size: 18pt;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def set_slide_page():
    st.set_page_config(
        page_title="Few-Shot Learning Benchmarks",
        layout="wide",
        page_icon=SICARA_LOGO,
    )

    set_theme()

    st.sidebar.markdown(
        """
    **Few-Shot Image Classification Benchmarks are Too Far From Reality: Build Back Better with Semantic Task Sampling** 
    
    Etienne Bennequin, Myriam Tami, Antoine Toubhans, Céline Hudelot
    """
    )

    # st.sidebar.image(SICARA_LOGO)


def vertical_space(height):
    components.html(
        f"""
    <div style="height: {height}px"></div>
    """,
        height=height,
    )


def navigation_buttons(key, n_steps):

    with st.sidebar:
        vertical_space(100)

    if st.session_state.get("intra_slide_step") is None:
        st.session_state.intra_slide_step = {}
    if st.session_state.intra_slide_step.get(key) is None:
        st.session_state.intra_slide_step[key] = 0
    if st.sidebar.button(" < "):
        st.session_state.intra_slide_step[key] = (
            st.session_state.intra_slide_step[key] - 1
            if st.session_state.intra_slide_step[key] > 0
            else 0
        )
    if st.sidebar.button(" > "):
        st.session_state.intra_slide_step[key] = (
            st.session_state.intra_slide_step[key] + 1
            if st.session_state.intra_slide_step[key] < n_steps
            else n_steps
        )

    with st.sidebar:
        components.html(
            """
        <script>
        const doc = window.parent.document  // break out of the IFrame
        const left_button = doc.querySelectorAll('button[kind=secondary]')[0]
        const right_button = doc.querySelectorAll('button[kind=secondary]')[1]
        doc.addEventListener('keyup', function (event) {
            if (event.key === 'p') {
                left_button.click()
            }
        });
        doc.addEventListener('keyup', function (event) {
            if (event.key === 'n') {
                right_button.click()
            }
        });
        </script>
        """,
            height=0,
            width=0,
        )
