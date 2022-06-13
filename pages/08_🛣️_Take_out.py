import streamlit as st

from st_scripts.st_utils.st_app_blocks import plot_semantic_graph
from st_scripts.st_utils.st_constants import set_slide_page, navigation_buttons
from st_scripts.st_utils.st_wordings import st_divider

set_slide_page()
key = 8

navigation_buttons(key, n_steps=8)

st.subheader("👀 The Few-Shot Learning community should look at its data")

if st.session_state.intra_slide_step[key] > 0:
    st_divider()
    st.subheader(
        "⚖ Use our semantic version *tiered*ImageNet for better balanced tasks"
    )

if st.session_state.intra_slide_step[key] > 1:
    st_divider()
    st.subheader("⏭️Next step: look at the chosen examples")
    st.image(
        "slides_images/08.png",
        use_column_width="always",
    )
