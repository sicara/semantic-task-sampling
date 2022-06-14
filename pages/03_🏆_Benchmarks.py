import streamlit as st

from st_scripts.st_utils.plot_helpers import render_svg
from st_scripts.st_utils.st_app_blocks import plot_semantic_graph
from st_scripts.st_utils.st_constants import set_slide_page, navigation_buttons

set_slide_page()

key = 3

navigation_buttons(key=3, n_steps=8)

render_svg(f"slides_images/03_{st.session_state.intra_slide_step[key]}.svg")
