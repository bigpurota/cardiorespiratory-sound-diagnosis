"""
tests/test_environment.py — EVAL-04 import and version assertions.

Tests that all required packages are importable and that pinned versions match
the project's pyproject.toml specification.

Wave 0: These tests will FAIL or SKIP until the uv environment is set up
(Phase 1, Wave 1). Collection must succeed with 0 errors.
"""
import importlib
import pytest

# ---------------------------------------------------------------------------
# Skip the whole module if torch is not importable (Wave 0 / pre-install state)
# ---------------------------------------------------------------------------
torch_spec = importlib.util.find_spec("torch")
_torch_missing = torch_spec is None


def test_imports():
    """All required packages import without error — EVAL-04."""
    packages = [
        "librosa",
        "torch",
        "torchaudio",
        "sklearn",
        "xgboost",
        "wfdb",
        "kaggle",
        "timm",
        "audiomentations",
        "soundfile",
        "seaborn",
    ]
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            assert mod is not None, f"import_module({pkg!r}) returned None"
        except ImportError:
            pytest.fail(
                f"package {pkg!r} not installed — run `uv sync` first to install all dependencies"
            )


@pytest.mark.skipif(
    _torch_missing,
    reason="torch not installed — run `uv sync` to set up the environment (Wave 0 skip)",
)
def test_versions():
    """Pinned versions match pyproject.toml — EVAL-04."""
    import torch  # noqa: E501
    import librosa  # noqa: E501
    import sklearn  # noqa: E501
    import xgboost  # noqa: E501

    assert torch.__version__.startswith("2.11"), (
        f"torch version mismatch: expected 2.11.x, got {torch.__version__}"
    )
    assert librosa.__version__ == "0.11.0", (
        f"librosa version mismatch: expected 0.11.0, got {librosa.__version__}"
    )
    assert sklearn.__version__ == "1.8.0", (
        f"scikit-learn version mismatch: expected 1.8.0, got {sklearn.__version__}"
    )
    assert xgboost.__version__.startswith("3.2"), (
        f"xgboost version mismatch: expected 3.2.x, got {xgboost.__version__}"
    )
