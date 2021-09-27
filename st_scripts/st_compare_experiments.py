import streamlit as st

from st_helpers import *

st.set_page_config(page_title="Compare experiments", layout="wide")

all_dvc_exps = get_all_exps()
all_metrics = get_metrics(all_dvc_exps.index.to_list())

selected_exps = st.sidebar.multiselect(
    label="Select experiments",
    options=all_dvc_exps.index,
    format_func=lambda x: f"{all_dvc_exps.parent_hash[x][:7]} - {x}",
)

st.title("Params")
all_params = get_params(selected_exps)


selected_params = st.multiselect(
    label="Select displayed params",
    options=all_params.columns.to_list(),
    default=all_params.columns.to_list(),
)

st.write(all_params[selected_params])

st.title("Metrics")
metrics_df = all_metrics.loc[lambda df: df.index.isin(selected_exps)]
st.write(metrics_df)


column_1, column_2 = st.columns(2)

bar_plot(metrics_df.accuracy, metrics_df["std"], column_1, title="TOP 1 accuracy")

st.title("Tensorboard")
url = download_tensorboards(
    revs=[all_dvc_exps.commit_hash.loc[exp] for exp in selected_exps],
)
# st.sidebar.write(url)
st.components.v1.iframe(url, height=900)
