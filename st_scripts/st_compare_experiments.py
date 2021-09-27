import streamlit as st

from st_helpers import get_all_exps, get_all_metrics, download_tensorboards, bar_plot

st.set_page_config(layout="wide")

all_dvc_exps = get_all_exps()
all_metrics = get_all_metrics(all_dvc_exps.index.to_list())

selected_exps = st.sidebar.multiselect(
    label="select commits",
    options=all_dvc_exps.index,
    format_func=lambda x: f"{all_dvc_exps.parent_hash[x][:7]} - {x}",
)

st.title("Metrics")
metrics_df = all_metrics.loc[lambda df: df.exp_name.isin(selected_exps)]
st.write(metrics_df)

column_1, column_2 = st.columns(2)

bar_plot(metrics_df.accuracy, metrics_df["std"], column_1, title="TOP 1 accuracy")

st.title("Tensorboard")
url = download_tensorboards(
    revs=[all_dvc_exps.commit_hash.loc[exp] for exp in selected_exps],
)
# st.sidebar.write(url)
st.components.v1.iframe(url, height=900)
