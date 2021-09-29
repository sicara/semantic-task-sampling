from st_helpers import *


def st_compare_experiments():

    all_dvc_exps = get_all_exps()
    all_metrics = get_metrics(all_dvc_exps.index.to_list())
    all_params = get_params(all_dvc_exps.index.to_list())

    selected_params = st.multiselect(
        label="Select displayed params",
        options=all_params.columns.to_list(),
        default=all_params.filter(regex=DEFAULT_DISPLAYED_PARAMS).columns.to_list(),
    )

    selected_exps = st.multiselect(
        label="Select experiments",
        options=all_dvc_exps.index,
        default=all_dvc_exps.index.to_list(),
        format_func=lambda x: display_fn(x, all_dvc_exps, all_params, selected_params),
    )
    st.title("Metrics")

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
        exps={
            exp: all_dvc_exps.commit_hash.loc[exp]
            for exp in all_dvc_exps.index.to_list()
        }
    )
    # st.sidebar.write(url)
    st.components.v1.iframe(url, height=900)
