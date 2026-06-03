"""Tests for the log-mel spectrogram helpers in"""
import importlib
import pathlib
import sys
import warnings

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

WINDOW_SAMPLES = 12000


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not implemented yet: {exc}")


def test_shape_dtype():
    """``window_to_logmel(window, make_mel(20,400))`` returns"""
    spectrograms = _import("src.spectrograms")
    for fn in ("make_mel", "window_to_logmel"):
        if not hasattr(spectrograms, fn):
            pytest.skip(f"src.spectrograms.{fn} not implemented yet")

    import numpy as np

    rng = np.random.default_rng(42)
    window = rng.standard_normal(WINDOW_SAMPLES).astype("float32")

    mel = spectrograms.make_mel(20, 400)
    spec = spectrograms.window_to_logmel(window, mel)

    spec = np.asarray(spec)
    assert spec.shape == (64, 128), f"expected (64,128) log-mel, got {spec.shape}"
    assert spec.dtype == np.float32, f"expected float32, got {spec.dtype}"


def test_no_filterbank_warning():
    """Building the heart 20–400 Hz mel (n_fft=512) emits NO"""
    spectrograms = _import("src.spectrograms")
    for fn in ("make_mel", "window_to_logmel"):
        if not hasattr(spectrograms, fn):
            pytest.skip(f"src.spectrograms.{fn} not implemented yet")

    import numpy as np

    rng = np.random.default_rng(42)
    window = rng.standard_normal(WINDOW_SAMPLES).astype("float32")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mel = spectrograms.make_mel(20, 400)
        spectrograms.window_to_logmel(window, mel)

    offenders = [
        str(w.message) for w in caught
        if "mel filterbank" in str(w.message).lower()
        or "all zero" in str(w.message).lower()
    ]
    assert not offenders, (
        "empty mel-filterbank warning emitted for the heart 20-400 Hz band — "
        f"make_mel must use n_fft=512. Got: {offenders}"
    )
