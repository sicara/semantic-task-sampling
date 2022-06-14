import streamlit as st

from st_scripts.st_utils.plot_helpers import render_svg
from st_scripts.st_utils.st_app_blocks import plot_semantic_graph
from st_scripts.st_utils.st_constants import set_slide_page

set_slide_page()


render_svg("slides_images/01.svg")
