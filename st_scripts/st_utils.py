import json

from hashlib import sha1
from typing import List

import streamlit as st


def aggregate_over_seeds(metrics_df, params, seed_column_name="train.seed"):
    return (
        metrics_df.groupby([param for param in params if param != seed_column_name])
        .aggregate(
            {
                **{
                    metric: "mean"
                    for metric in metrics_df.columns
                    if metric not in params
                },
                seed_column_name: "count",
            }
        )
        .rename(columns={seed_column_name: "n_seeds"})
    )


@st.cache
def condense_results(results):
    return (
        results.sort_values("score", ascending=False)
        .drop_duplicates(["task_id", "image_id"])
        .sort_values(["task_id", "image_id"])
        .reset_index(drop=True)
        .assign(accuracy=lambda df: df.true_label == df.predicted_label)
        .groupby(["task_id", "true_label"])
        .accuracy.mean()
    )


def get_hash_from_list(list_to_hash: List) -> str:
    return sha1(json.dumps(sorted(list_to_hash)).encode()).hexdigest()
