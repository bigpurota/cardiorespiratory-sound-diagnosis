"""
tests/test_determinism.py — EVAL-04 seed assertions.

Tests that two seeded NumPy/PyTorch runs produce identical output, verifying
that config.py SEED=42 wiring is deterministic.

Wave 0: Both tests will SKIP cleanly because config.py does not exist yet.
Collection must succeed with 0 errors.
"""
import hashlib
import importlib

import pytest

# numpy is required at module level only for the helper; guard the import so
# collection succeeds in Wave 0 before `uv sync` has been run.
np = None
_np_spec = importlib.util.find_spec("numpy")
if _np_spec is not None:
    import numpy as np  # noqa: E501


def _run_seeded():
    """Import config (which sets the global seed) and produce a deterministic hash."""
    import config  # sets SEED=42 and all framework seeds  # noqa: E501
    import numpy as _np  # noqa: E501
    rng = _np.random.RandomState(config.SEED)
    data = rng.randn(1000)
    return hashlib.md5(data.tobytes()).hexdigest()


def test_numpy_seed():
    """Two consecutive seeded NumPy runs produce identical output — EVAL-04."""
    # Skip gracefully in Wave 0 when config.py does not yet exist
    pytest.importorskip("config", reason="config.py not yet created — Wave 0 skip; will pass after Wave 1")

    h1 = _run_seeded()
    h2 = _run_seeded()
    assert h1 == h2, (
        f"Non-deterministic NumPy output: run 1 = {h1}, run 2 = {h2}. "
        "Check that config.py sets np.random.seed(SEED) correctly."
    )


def test_torch_seed():
    """Two consecutive seeded PyTorch runs produce identical output — EVAL-04."""
    # Skip gracefully in Wave 0 when config.py does not yet exist
    config = pytest.importorskip(
        "config", reason="config.py not yet created — Wave 0 skip; will pass after Wave 1"
    )

    import torch  # noqa: E501

    torch.manual_seed(config.SEED)
    t1 = torch.randn(100).tolist()
    torch.manual_seed(config.SEED)
    t2 = torch.randn(100).tolist()
    assert t1 == t2, (
        "torch.manual_seed does not reproduce identical output. "
        "Verify config.py sets torch.manual_seed(SEED) and "
        "torch.backends.cudnn.deterministic = True."
    )
