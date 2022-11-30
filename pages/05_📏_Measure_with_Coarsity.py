import pandas as pd
import streamlit as st

from st_scripts.st_utils.data_fetchers import (
    get_testbed,
    get_easyset_expo,
    get_class_names,
)
from st_scripts.st_utils.st_app_blocks import plot_semantic_graph
from st_scripts.st_utils.st_constants import set_slide_page

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

st.title("Measuring the problem")

with st.expander("Jiang & Conrath", expanded=True):
    st.markdown(
        """
    On a Directed Acyclic Graph (DAG), we can use the 
    Jiang & Conrath pseudo-distance between any two classes $$c_1$$ and $$c_2$$:
    """
    )

    st.latex(
        r"""
        D^{JC}(c_1, c_2) = 2 \log |lso(c_1,c_2)| - (\log |c_1| + \log |c_2|)
    """
    )

    st.markdown(
        r"""
    where $$|c|$$ is the number of instances of the dataset with class $$c$$, 
    and $$lso(c_1, c_2)$$ is their most specific common ancestor.
    """
    )


with st.expander("Coarsity", expanded=False):
    st.markdown(
        r"""
    From this, we define the coarsity $$\kappa$$ of a task 
    $$\mathbf T_{\mathbf C}$$ constituted of instances 
    from a set of classes $$\mathbf C$$ :
    """
    )

    st.latex(
        r"""
           \kappa(\mathbf T_{\mathbf C}) = \underset{c_i, c_j \in \mathbf C ;~ 
c_i \ne c_j}{\text{mean}}  D^{JC}(c_i, c_j)^2
    """
    )

with st.expander("tieredImageNet"):
    st.markdown("*tiered*ImageNet's classes are part of the WordNet graph!")
    graph_column, distances_column = st.columns([3, 3])
    with graph_column:
        plot_semantic_graph()

    with distances_column:
        vertical_space(45)

        st.latex(
            r"""
            D^{JC}(\textcolor{red}{hotdog}, \textcolor{blue}{cheeseburger}) = 1.39 \\
        """
        )
        vertical_space(25)
        st.latex(
            r"""
            D^{JC}(\textcolor{red}{hotdog}, \textcolor{green}{guacamole}) = 6.06 \\
        """
        )
        vertical_space(25)
        st.latex(
            r"""
            D^{JC}(\textcolor{red}{hotdog}, \textcolor{gold}{goldfish}) = 10.13 \\
        """
        )
