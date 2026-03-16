"""
tests/test_features.py — DATA-05 feature-extraction contracts (Phase 3, Wave 0, RED).

Specifies the contracts for the Wave-1 pure functions described in
03-RESEARCH.md §Pattern 1 (heart window → Set A 240-d / Set B 250-d) and
§Pattern 2 (lung cycle PAD-BEFORE-EXTRACT to 3.0 s so ``librosa.feature.delta``
does not raise ``ParameterError`` on short cycles):

  - src.features.window_feature_vector(w, sr=4000, include_spectral=False)
      MFCC(n_mfcc=40) + Δ + ΔΔ summarised as mean+std → 240-d (Set A);
      include_spectral=True adds 5 spectral stats × (mean,std) = 10 → 250-d (Set B).
  - src.features.lung_cycle_vector(yb, start_s, end_s, sr=4000, pad_s=3.0, ...)
      slices [start_s, end_s], pads/trims to 3.0 s (12000 samples) BEFORE MFCC so
      a 0.2-s cycle (which raw would yield only 2 frames and crash delta) succeeds.

These reference src.features symbols that do NOT exist yet, so the tests MUST be
RED now. Collection MUST succeed: the import happens INSIDE each test body
(skip-on-missing) so a missing module never errors at collection time. This
mirrors tests/test_preprocess.py exactly.
"""
import importlib
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity

# Feature-vector dimension contract (VERIFIED live in 03-RESEARCH.md §Pattern 1).
DIM_SET_A = 240   # 6 * 40 = (mean+std) × (MFCC, Δ, ΔΔ), n_mfcc=40
DIM_SET_B = 250   # Set A + 5 spectral stats × (mean, std)
FS = 4000
HEART_WINDOW = 12000   # int(3.0 * 4000)


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent in Wave 0."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive (Wave 0 module absent)
        pytest.skip(f"{module_name} not implemented yet (Wave 0): {exc}")


def _synthetic_window(n=HEART_WINDOW, seed=42):
    """Deterministic 3-s heart-window-like array (sine + low noise), float32.

    No downloaded data: stands in for a real 3-s window when exercising the
    pure feature-extraction math.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64) / FS
    tone = np.sin(2.0 * np.pi * 100.0 * t)        # cardiac-band tone
    noise = 0.05 * rng.standard_normal(n)
    return (tone + noise).astype("float32")


# ---------------------------------------------------------------------------
# Heart window feature-vector dimensions — 240 (Set A) / 250 (Set B)
# ---------------------------------------------------------------------------

def test_heart_vector_dims():
    """A 12000-sample heart window yields a 240-d (Set A) / 250-d (Set B) float32 vector."""
    features = _import("src.features")
    if not hasattr(features, "window_feature_vector"):
        pytest.skip("src.features.window_feature_vector not implemented yet (Wave 0)")

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


# ---------------------------------------------------------------------------
# Lung short cycle — PAD BEFORE EXTRACT (no ParameterError on shortest cycle)
# ---------------------------------------------------------------------------

def test_lung_short_cycle_pad():
    """A 0.2-s lung cycle is padded to 3.0 s BEFORE MFCC → delta succeeds, finite vector.

    Raw, a 0.2-s cycle yields only 2 MFCC frames and ``librosa.feature.delta``
    (default width=9) raises ParameterError. The pad-before-extract guarantee
    (03-RESEARCH.md §Pattern 2 / Pitfall 2) must make this succeed.
    """
    features = _import("src.features")
    if not hasattr(features, "lung_cycle_vector"):
        pytest.skip("src.features.lung_cycle_vector not implemented yet (Wave 0)")

    # A SHORT cycle: 0.2 s = 800 samples, embedded in a slightly longer buffer.
    rng = np.random.default_rng(7)
    short_n = int(0.2 * FS)                       # 800 samples
    buf = (np.sin(2.0 * np.pi * 300.0 * np.arange(short_n, dtype=np.float64) / FS)
           + 0.05 * rng.standard_normal(short_n)).astype("float32")

    # Must NOT raise ParameterError (pad-before-delta); returns a finite Set A vector.
    vec = np.asarray(
        features.lung_cycle_vector(buf, start_s=0.0, end_s=0.2, sr=FS, include_spectral=False)
    )
    assert vec.shape == (DIM_SET_A,), (
        f"padded lung cycle vector must be ({DIM_SET_A},); got {vec.shape}"
    )
    assert np.all(np.isfinite(vec)), "padded lung cycle vector contains NaN/Inf"


# ---------------------------------------------------------------------------
# Vectors are NaN/Inf free (T-03-V5 data-integrity threat)
# ---------------------------------------------------------------------------

def test_vectors_nan_free():
    """Heart and lung feature vectors are entirely finite (no NaN / Inf)."""
    features = _import("src.features")
    if not (hasattr(features, "window_feature_vector") and hasattr(features, "lung_cycle_vector")):
        pytest.skip("src.features feature functions not implemented yet (Wave 0)")

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
