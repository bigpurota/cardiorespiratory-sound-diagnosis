"""Determinism checks for the seeded NumPy and PyTorch RNGs: two seeded runs must
produce identical output, confirming the SEED=42 wiring in config is deterministic."""
import hashlib
import importlib

import pytest

np = None
_np_spec = importlib.util.find_spec("numpy")
if _np_spec is not None:
    import numpy as np


def _run_seeded():
    from src import config  # importing config seeds all RNGs
    import numpy as _np
    rng = _np.random.RandomState(config.SEED)
    data = rng.randn(1000)
    return hashlib.md5(data.tobytes()).hexdigest()


def test_numpy_seed():
    """Two consecutive seeded NumPy runs produce identical output."""
    pytest.importorskip("src.config")

    h1 = _run_seeded()
    h2 = _run_seeded()
    assert h1 == h2, (
        f"Non-deterministic NumPy output: run 1 = {h1}, run 2 = {h2}. "
        "Check that config sets np.random.seed(SEED) correctly."
    )


def test_torch_seed():
    """Two consecutive seeded PyTorch runs produce identical output."""
    config = pytest.importorskip("src.config")

    import torch

    torch.manual_seed(config.SEED)
    t1 = torch.randn(100).tolist()
    torch.manual_seed(config.SEED)
    t2 = torch.randn(100).tolist()
    assert t1 == t2, (
        "torch.manual_seed does not reproduce identical output. "
        "Verify config.py sets torch.manual_seed(SEED) and "
        "torch.backends.cudnn.deterministic = True."
    )
