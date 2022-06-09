import streamlit as st

from st_scripts.st_utils.st_constants import set_slide_page

set_slide_page()

st.title("Measuring the problem")

col1, col2 = st.columns([3, 2])

with col1:
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

with col2:
    with st.expander("tieredImageNet"):
        st.markdown("*tiered*ImageNet's classes are part of the WordNet graph!")
