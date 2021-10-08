from pathlib import Path

from dvc.repo import Repo

DVC_REPO = Repo("")
PARAMS_FILE = "params.yaml"
METRICS_DIR = Path("data/tiered_imagenet/metrics")
METRICS_FILE = METRICS_DIR / "evaluation_metrics.json"
TENSORBOARD_LOGS_DIR = Path("data/tiered_imagenet/tb_logs")
TENSORBOARD_CACHE_DIR = Path("streamlit_cache") / "tensorboard"
DEFAULT_DISPLAYED_PARAMS = "train.*"
