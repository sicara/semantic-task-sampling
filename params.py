from pathlib import Path

root_dir = Path("data/tiered_imagenet")

specs_dir = root_dir / "specs"
distances_dir = root_dir / "distances"
models_dir = root_dir / "models"
metrics_dir = root_dir / "metrics"
tb_logs_dir = root_dir / "tb_logs"
logs_dir = root_dir / "logs"
testbeds_dir = root_dir / "testbeds"

alpha_grid = [
    0.005,
    0.01,
    0.05,
    0.1,
]

grid_search = [
    {"sampler": "semantic", "alpha": alpha, "name": f"semantic_{alpha}"}
    for alpha in alpha_grid
] + [
    {
        "sampler": "uniform",
        "alpha": None,
        "name": "uniform",
    },
    {
        "sampler": "adaptive",
        "alpha": None,
        "name": "adaptive",
    },
]
