import json
from hashlib import sha1
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import dvc.api

from dvc.repo import Repo
from dvc.repo.get import get
from streamlit.delta_generator import DeltaGenerator
from tensorboard import program

DVC_REPO = Repo(".")


def read_metrics(rev: str) -> Dict:
    with dvc.api.open(
            "data/tiered_imagenet/metrics/evaluation_metrics.json",
            rev=rev
    ) as file:
        evaluation_metrics = json.load(file)

    return evaluation_metrics


@st.cache
def get_all_exps():
    all_dvc_exps_dict = DVC_REPO.experiments.ls(all_=True)
    return pd.DataFrame(
        [
            (exp_name, parent_commit, DVC_REPO.experiments.scm.resolve_rev(exp_name))
            for parent_commit in all_dvc_exps_dict
            for exp_name in all_dvc_exps_dict[parent_commit]
        ], columns=["exp_name", "parent_hash", "commit_hash"]
    ).set_index("exp_name")

ALL_DVC_EXPS = get_all_exps()
st.write([exp for exp in ALL_DVC_EXPS.index])
@st.cache
def get_all_metrics():
    return pd.DataFrame([
        {
            "exp_name": exp,
            **read_metrics(exp)
        }
            for exp in ALL_DVC_EXPS.index
    ])

TENSORBOARD_CACHE_DIR = Path("streamlit_cache") / "tensorboard"

def get_hash_from_list(list_to_hash: List) -> str:
    return sha1(json.dumps(sorted(list_to_hash)).encode()).hexdigest()

def download_dir(path, git_rev, out):
    get(
        url=".", path=path, out=out, rev=git_rev,
    )

def download_tensorboards(revs):
    cache_dir = TENSORBOARD_CACHE_DIR / get_hash_from_list(revs)
    # if not cache_dir.exists():
    for git_rev in revs:
        download_dir(
            path="data/tiered_imagenet/tb_logs",
            git_rev="ceci n'existe pas",
            out=str(cache_dir / git_rev)
        )

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(cache_dir)])
    return tb.launch()

def bar_plot(accuracy: pd.Series, error: pd.Series, st_column: DeltaGenerator, title: str):

    fig, ax = plt.subplots()
    accuracy.plot.bar(
        ax=ax, title=title, fontsize=15, alpha=0.5, grid=True, yerr=error,
    )
    plt.legend(loc="lower left")
    st_column.pyplot(fig)


st.write(ALL_DVC_EXPS)
selected_exps = st.sidebar.multiselect(
    label="select commits",
    options=ALL_DVC_EXPS.index,
    format_func=lambda x: f"{ALL_DVC_EXPS.parent_hash[x][:7]} - {x}",
)


# all_metrics = get_all_metrics()
# metrics_df = all_metrics.loc[lambda df: df.index.isin(selected_exps)]
# st.write(all_metrics)
#
# column_1, column_2 = st.columns(2)
#
# bar_plot(metrics_df.accuracy, metrics_df["std"], column_1, title="TOP 1 accuracy")

url = download_tensorboards(
    revs=selected_exps,
)
# st.sidebar.write(url)
# st.components.v1.iframe(url, height=900)
