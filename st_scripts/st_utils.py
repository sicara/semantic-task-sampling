import json

from hashlib import sha1
from typing import List


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


def get_hash_from_list(list_to_hash: List) -> str:
    return sha1(json.dumps(sorted(list_to_hash)).encode()).hexdigest()
