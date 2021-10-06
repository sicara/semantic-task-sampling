import json

import yaml
from PIL import Image
from dvc.repo.get import get
from hashlib import sha1
from pathlib import Path
from typing import Dict, List

import dvc.api
import pandas as pd
import streamlit as st
from dvc.repo import Repo
from matplotlib import pyplot as plt
from streamlit.delta_generator import DeltaGenerator
from tensorboard import program

DVC_REPO = Repo("")
PARAMS_FILE = "params.yaml"
METRICS_DIR = Path("data/tiered_imagenet/metrics")
METRICS_FILE = METRICS_DIR / "evaluation_metrics.json"
TENSORBOARD_LOGS_DIR = Path("data/tiered_imagenet/tb_logs")
TENSORBOARD_CACHE_DIR = Path("streamlit_cache") / "tensorboard"
DEFAULT_DISPLAYED_PARAMS = "train.*"


@st.cache
def read_params(rev: str) -> Dict:
    with dvc.api.open(PARAMS_FILE, rev=rev) as file:
        params = yaml.safe_load(file)

    return params


@st.cache
def read_metrics(rev: str) -> Dict:
    with dvc.api.open(METRICS_FILE, rev=rev) as file:
        evaluation_metrics = json.load(file)

    return evaluation_metrics


def get_params(
    exp_list: List[str],
) -> pd.DataFrame:
    return pd.concat(
        [
            pd.json_normalize(read_params(exp), sep=".").assign(exp_name=exp)
            for exp in exp_list
        ]
    ).set_index("exp_name")


def get_metrics(exp_list: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"exp_name": exp, **read_metrics(exp)} for exp in exp_list]
    ).set_index("exp_name")


@st.cache
def get_all_exps():
    all_dvc_exps_dict = DVC_REPO.experiments.ls(all_=True)
    return pd.DataFrame(
        [
            (exp_name, parent_commit, DVC_REPO.experiments.scm.resolve_rev(exp_name))
            for parent_commit in all_dvc_exps_dict
            for exp_name in all_dvc_exps_dict[parent_commit]
        ],
        columns=["exp_name", "parent_hash", "commit_hash"],
    ).set_index("exp_name")


def plot_all_bars(metrics_df):
    bar_plots_columns = st.columns(5)

    with bar_plots_columns[0]:
        bar_plot(metrics_df.accuracy, title="TOP 1 overall accuracy")

    for i, quartile in enumerate(["first", "second", "third", "fourth"]):
        with bar_plots_columns[i + 1]:
            bar_plot(
                metrics_df[f"{quartile}_quartile_acc"],
                title=f"TOP 1 accuracy on {quartile} quartile",
            )


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


def download_dir(path, git_rev, out):
    get(
        url=".",
        path=path,
        out=out,
        rev=git_rev,
    )


@st.cache
def download_tensorboards(exps: Dict[str, str]):
    cache_dir = TENSORBOARD_CACHE_DIR / get_hash_from_list(list(exps.keys()))
    if not cache_dir.exists():
        for exp, git_rev in exps.items():
            download_dir(
                path=str(TENSORBOARD_LOGS_DIR),
                git_rev=git_rev,
                out=str(cache_dir / exp),
            )

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(cache_dir)])
    return tb.launch()


def bar_plot(accuracy: pd.Series, title: str):
    bottom = 0.9 * accuracy.min()

    fig, ax = plt.subplots()
    (accuracy - bottom).plot.bar(
        ax=ax,
        title=title,
        fontsize=15,
        alpha=0.5,
        grid=True,
        bottom=bottom,
    )
    # plt.legend(loc="lower left")
    st.pyplot(fig)


def plot_image(path: Path, exp: str, caption: str = None):
    with dvc.api.open(path, rev=exp, mode="rb") as file:
        st.image(Image.open(file), caption=caption)


def read_csv(path: Path, exp: str) -> pd.DataFrame:
    with dvc.api.open(path, rev=exp, mode="r") as file:
        df = pd.read_csv(file, index_col=0)
    return df


def display_fn(x, exps_df, all_params, selected_params):
    to_display = f"{exps_df.parent_hash[x][:7]} - {x}"
    for param in selected_params:
        to_display += f" - {param} {all_params[param][x]}"
    return to_display
