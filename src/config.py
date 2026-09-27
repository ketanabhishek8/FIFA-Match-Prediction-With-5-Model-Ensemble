"""Central config loading + reproducibility helpers.

Usage:
    from src.config import load_config, set_seed

    cfg = load_config()
    set_seed(cfg["project"]["random_state"])
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML config file into a plain dict."""
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed python's random module and numpy for reproducibility.

    Call this once at the start of any training/eval script, e.g.:
        set_seed(cfg["project"]["random_state"])
    """
    random.seed(seed)
    np.random.seed(seed)


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a config-relative path (e.g. 'data/raw') against PROJECT_ROOT."""
    return PROJECT_ROOT / relative_path
