"""Where generated datasets and results live.

Defaults to ``data/`` and ``results/`` in the repo (both git-ignored). Set
``JAMLOC_DATA_DIR`` / ``JAMLOC_RESULTS_DIR`` to put them elsewhere, e.g. on Google
Drive when running on Colab, where the VM's disk is wiped after each session.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _dir(env_var: str, default: Path) -> Path:
    path = Path(os.environ.get(env_var, default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    return _dir("JAMLOC_DATA_DIR", REPO_ROOT / "data")


def results_dir() -> Path:
    return _dir("JAMLOC_RESULTS_DIR", REPO_ROOT / "results")
