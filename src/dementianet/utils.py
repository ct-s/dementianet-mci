"""Shared utilities: config loading, seeding, path resolution."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "config/config.yaml") -> dict:
    """Load the YAML config relative to the project root."""
    cfg_path = PROJECT_ROOT / path
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Seed Python and NumPy RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path to an absolute path under the project root."""
    return PROJECT_ROOT / path
