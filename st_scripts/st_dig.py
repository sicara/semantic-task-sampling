import matplotlib.pyplot as plt

import streamlit as st

from st_helpers import (
    get_all_exps,
    read_metrics,
    read_params,
    METRICS_DIR,
    plot_image,
    read_csv,
    get_params,
    DEFAULT_DISPLAYED_PARAMS,
    display_fn,
)


def st_dig():
    st.title("Dig an experiment")

    all_dvc_exps = get_all_exps()
    all_params = get_params(all_dvc_exps.index.to_list())

    selected_params = st.multiselect(
        label="Select displayed params",
        options=all_params.columns.to_list(),
        default=all_params.filter(regex=DEFAULT_DISPLAYED_PARAMS).columns.to_list(),
    )

    selected_exp = st.selectbox(
        label="Select an experiment",
        options=all_dvc_exps.index,
        format_func=lambda x: display_fn(x, all_dvc_exps, all_params, selected_params),
    )

    metrics = read_metrics(selected_exp)
    params = read_params(selected_exp)

    columns = st.columns(3)

    columns[0].write(metrics)
    columns[0].write(params)

    plot_image(
        METRICS_DIR / "training_classes_biconfusion.png",
        selected_exp,
        columns[1],
        caption="Training classes biconfusion",
    )
    plot_image(
        METRICS_DIR / "training_classes_sampled_together.png",
        selected_exp,
        columns[1],
        caption="Training classes cosampling",
    )

    fig, ax = plt.subplots()
    read_csv(METRICS_DIR / "task_performances.csv", selected_exp).plot.scatter(
        x="variance",
        y="accuracy",
        ax=ax,
        title="Accuracy depending on intra-task distance on test set",
    )
    columns[2].pyplot(fig)

    fig, ax = plt.subplots()
    intra_training_task_distances = read_csv(
        METRICS_DIR / "intra_training_task_distances.csv", selected_exp
    ).assign(smooth=lambda df: df.median_distance.rolling(500).mean())

    intra_training_task_distances.smooth.plot.line(
        ax=ax, title="Evolution of intra-task distances during training (smooth)"
    )
    columns[2].pyplot(fig)
