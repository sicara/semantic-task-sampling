import streamlit as st

from st_scripts.st_utils.st_app_blocks import plot_semantic_graph
from st_scripts.st_utils.st_constants import set_slide_page, vertical_space

set_slide_page()

col1, col2 = st.columns([2, 2])

with col1:
    st.image("slides_images/QR.png", use_column_width="always")

with col2:
    vertical_space(30)
    st.header("Our paper")

    st.markdown(
        """
    📃 [Paper](https://arxiv.org/abs/2205.05155)
    
    💿 [Code](https://github.com/sicara/semantic-task-sampling)
    
    🌐 [Project website](https://share.streamlit.io/sicara/semantic-task-sampling)
    """
    )

    st.header("Contact")

    st.markdown(
        """
    ✉️ [etienneb@sicara.com](mailto:etienneb@sicara.com)
    
    🐦 [@EBennequin](https://twitter.com/EBennequin)
    
    🏠 [Personal page](https://ebennequin.github.io/)
    """
    )

with st.expander("References"):
    st.markdown(
        """
    [Transductive information maximization for few-shot learning](https://proceedings.neurips.cc/paper/2020/hash/196f5641aa9dc87067da4ff90fd81e7b-Abstract.html). Boudiaf et al., 2020
    
    [A closer look at few-shot classification](https://arxiv.org/pdf/1904.04232.pdf). Chen et al., 2019
    
    [A baseline for few-shot image classification](https://arxiv.org/pdf/1909.02729.pdf). Dhillon et al., 2020
    
    [Leveraging the feature distribution in transfer-based few-shot learning](https://arxiv.org/pdf/2006.03806.pdf). Hu et al., 2021
    
    [Semantic similarity based on corpus statistics and lexical taxonomy](https://arxiv.org/pdf/cmp-lg/9709008.pdf). Jiang & Conrath, 1997
    
    [Adaptive task sampling for meta-learning](https://arxiv.org/pdf/2007.08735.pdf). Liu et al., 2020
    
    [Prototype rectification for few-shot learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460715.pdf). Liu et al., 2020
    
    [Wordnet: a lexical database for english](https://dl.acm.org/doi/pdf/10.1145/219717.219748). George A. Miller, 1995
    
    [Imagenet: A large-scale hierarchical image database](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5206848). Deng et al., 2009
    
    [Matching networks for one shot learning](https://proceedings.neurips.cc/paper/2016/file/90e1357833654983612fb05e3ec9148c-Paper.pdf). Vinyals et al., 2016
    
    [Meta-learning for semi-supervised few-shot classification](https://arxiv.org/pdf/1803.00676.pdf). Ren et al., 2019
    
    [Prototypical networks for few-shot learning](https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html). Snell et al., 2017
    """
    )
