"""Feature-extraction contracts for the pure functions in"""
import importlib
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

DIM_SET_A = 240
DIM_SET_B = 250
FS = 4000
HEART_WINDOW = 12000


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if it is"""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not implemented yet: {exc}")


def _synthetic_window(n=HEART_WINDOW, seed=42):
    """Deterministic 3-s heart-window-like array (sine + low"""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / FS
    tone = np.sin(2.0 * np.pi * 100.0 * t)
    noise = 0.05 * rng.standard_normal(n)
    return (tone + noise).astype("float32")


def test_heart_vector_dims():
    """A 12000-sample heart window yields a 240-d (Set A) / 250-d"""
    features = _import("src.features")
    if not hasattr(features, "window_feature_vector"):
        pytest.skip("src.features.window_feature_vector not implemented yet")

    w = _synthetic_window()

    vec_a = np.asarray(features.window_feature_vector(w, sr=FS, include_spectral=False))
    assert vec_a.shape == (DIM_SET_A,), (
        f"Set A window vector must be ({DIM_SET_A},); got {vec_a.shape}"
    )
    assert vec_a.dtype == np.float32, f"Set A vector must be float32, got {vec_a.dtype}"

    vec_b = np.asarray(features.window_feature_vector(w, sr=FS, include_spectral=True))
    assert vec_b.shape == (DIM_SET_B,), (
        f"Set B window vector must be ({DIM_SET_B},); got {vec_b.shape}"
    )


def test_lung_short_cycle_pad():
    """A 0.2-s lung cycle is padded to 3.0 s BEFORE MFCC → delta"""
    features = _import("src.features")
    if not hasattr(features, "lung_cycle_vector"):
        pytest.skip("src.features.lung_cycle_vector not implemented yet")

    rng = np.random.default_rng(7)
    short_n = int(0.2 * FS)
    buf = (np.sin(2.0 * np.pi * 300.0 * np.arange(short_n, dtype=np.float64) / FS)
           + 0.05 * rng.standard_normal(short_n)).astype("float32")

    vec = np.asarray(
        features.lung_cycle_vector(buf, start_s=0.0, end_s=0.2, sr=FS, include_spectral=False)
    )
    assert vec.shape == (DIM_SET_A,), (
        f"padded lung cycle vector must be ({DIM_SET_A},); got {vec.shape}"
    )
    assert np.all(np.isfinite(vec)), "padded lung cycle vector contains NaN/Inf"


def test_vectors_nan_free():
    """Heart and lung feature vectors are entirely finite (no NaN"""
    features = _import("src.features")
    if not (hasattr(features, "window_feature_vector") and hasattr(features, "lung_cycle_vector")):
        pytest.skip("src.features feature functions not implemented yet")

    w = _synthetic_window()
    heart_vec = np.asarray(features.window_feature_vector(w, sr=FS, include_spectral=True))
    assert np.all(np.isfinite(heart_vec)), "heart feature vector contains NaN/Inf"

    rng = np.random.default_rng(11)
    short_n = int(0.2 * FS)
    buf = (np.sin(2.0 * np.pi * 300.0 * np.arange(short_n, dtype=np.float64) / FS)
           + 0.05 * rng.standard_normal(short_n)).astype("float32")
    lung_vec = np.asarray(
        features.lung_cycle_vector(buf, start_s=0.0, end_s=0.2, sr=FS, include_spectral=True)
    )
    assert np.all(np.isfinite(lung_vec)), "lung feature vector contains NaN/Inf"
