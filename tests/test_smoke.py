"""Smoke tests -- run with `pytest`. These check the scaffold wires together,
not model quality. Expand as you implement each module.
"""
import numpy as np

from dementianet.utils import load_config, set_seed


def test_config_loads():
    cfg = load_config()
    assert "audio" in cfg
    assert cfg["audio"]["target_sample_rate"] == 16000


def test_seed_is_deterministic():
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert np.allclose(a, b)
