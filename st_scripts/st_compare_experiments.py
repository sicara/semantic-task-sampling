import streamlit as st

from st_displayers import (
    display_fn,
    plot_all_bars,
)
from dvc_getters import (
    get_params,
    get_metrics,
    get_all_exps,
    download_tensorboards,
    DEFAULT_DISPLAYED_PARAMS,
)
from st_utils import aggregate_over_seeds


def st_compare_experiments():

    all_dvc_exps = get_all_exps()
    all_metrics = get_metrics(all_dvc_exps.index.to_list())
    all_params = get_params(all_dvc_exps.index.to_list())

    st.title("Selection")

    selected_params = st.multiselect(
        label="Select displayed params",
        options=all_params.columns.to_list(),
        default=all_params.filter(regex=DEFAULT_DISPLAYED_PARAMS).columns.to_list(),
    )

    selected_commits = st.multiselect(
        "Select groups of experiments by commits",
        options=all_dvc_exps.parent_hash.unique(),
        format_func=lambda x: x[:7]
        + f" ({all_dvc_exps.parent_hash.value_counts().loc[x]} experiments)",
    )

    selected_exps = st.multiselect(
        label="Select experiments",
        options=all_dvc_exps.index,
        default=all_dvc_exps.loc[
            lambda df: df.parent_hash.isin(selected_commits)
        ].index.to_list(),
        format_func=lambda x: display_fn(x, all_dvc_exps, all_params, selected_params),
    )

    if len(selected_exps) > 0:

        with st.expander("See params as JSON"):
            st.write(
                all_params[selected_params]
                .loc[lambda df: df.index.isin(selected_exps)]
                .to_dict(orient="index")
            )

        st.title("Metrics")

        metrics_df = (
            all_params[selected_params]
            .join(all_metrics)
            .loc[lambda df: df.index.isin(selected_exps)]
        )
        st.write(metrics_df)

        plot_all_bars(metrics_df)

        with st.expander("Aggregate over seeds"):
            metrics_over_seeds_df = aggregate_over_seeds(metrics_df, selected_params)

            st.write(metrics_over_seeds_df)

            plot_all_bars(metrics_over_seeds_df)

    st.title("Tensorboard")

    url = download_tensorboards(
        exps={
            exp: all_dvc_exps.commit_hash.loc[exp]
            for exp in all_dvc_exps.index.to_list()
        }
    )
    # st.sidebar.write(url)
    st.components.v1.iframe(url, height=900)
