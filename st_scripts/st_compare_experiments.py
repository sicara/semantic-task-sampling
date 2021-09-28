from st_helpers import *

st.set_page_config(page_title="Compare experiments", layout="wide")

all_dvc_exps = get_all_exps()
all_metrics = get_metrics(all_dvc_exps.index.to_list())
all_params = get_params(all_dvc_exps.index.to_list())

selected_exps = st.sidebar.multiselect(
    label="Select experiments",
    options=all_dvc_exps.index,
    default=all_dvc_exps.index.to_list(),
    format_func=lambda x: f"{all_dvc_exps.parent_hash[x][:7]} - {x}",
)

st.title("Metrics")

selected_params = st.multiselect(
    label="Select displayed params",
    options=all_params.columns.to_list(),
    default=all_params.filter(regex=DEFAULT_DISPLAYED_PARAMS).columns.to_list(),
)

metrics_df = (
    all_params[selected_params]
    .join(all_metrics)
    .loc[lambda df: df.index.isin(selected_exps)]
)
st.write(metrics_df)


columns = st.columns(5)

bar_plot(metrics_df.accuracy, columns[0], title="TOP 1 overall accuracy")

for i, quartile in enumerate(["first", "second", "third", "fourth"]):
    bar_plot(
        metrics_df[f"{quartile}_quartile_acc"],
        columns[i + 1],
        title=f"TOP 1 accuracy on {quartile} quartile",
    )

st.title("Tensorboard")
url = download_tensorboards(
    exps={exp: all_dvc_exps.commit_hash.loc[exp] for exp in selected_exps}
)
# st.sidebar.write(url)
st.components.v1.iframe(url, height=900)
