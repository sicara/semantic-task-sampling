import json

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
METRICS_FILE = "data/tiered_imagenet/metrics/evaluation_metrics.json"
TENSORBOARD_CACHE_DIR = Path("streamlit_cache") / "tensorboard"
TENSORBOARD_LOGS_DIR = "data/tiered_imagenet/tb_logs"

def read_metrics(rev: str) -> Dict:
    with dvc.api.open(
        METRICS_FILE, rev=rev
    ) as file:
        evaluation_metrics = json.load(file)

    return evaluation_metrics

@st.cache
def get_all_metrics(exp_list: List[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"exp_name": exp, **read_metrics(exp)} for exp in exp_list]
    )


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
def download_tensorboards(revs):
    cache_dir = TENSORBOARD_CACHE_DIR / get_hash_from_list(revs)
    if not cache_dir.exists():
        for git_rev in revs:
            download_dir(
                path=TENSORBOARD_LOGS_DIR,
                git_rev=git_rev,
                out=str(cache_dir / git_rev),
            )

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", str(cache_dir)])
    return tb.launch()


def bar_plot(
    accuracy: pd.Series, error: pd.Series, st_column: DeltaGenerator, title: str
):

    fig, ax = plt.subplots()
    accuracy.plot.bar(
        ax=ax,
        title=title,
        fontsize=15,
        alpha=0.5,
        grid=True,
        yerr=error,
    )
    plt.legend(loc="lower left")
    st_column.pyplot(fig)
