"""Import and version checks for the project's pinned dependencies.

Verifies that all required packages are importable and that pinned versions match
the project's pyproject.toml specification.
"""
import importlib
import pytest

# ---------------------------------------------------------------------------
# Skip the whole module if torch is not importable (pre-install state)
# ---------------------------------------------------------------------------
torch_spec = importlib.util.find_spec("torch")
_torch_missing = torch_spec is None


def test_imports():
    """All required packages import without error."""
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
    reason="torch not installed — run `uv sync` to set up the environment",
)
def test_versions():
    """Pinned versions match pyproject.toml."""
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
