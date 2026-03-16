"""
Shared fixtures and constants for the DSBA cardiorespiratory sound project test suite.

This file is collected automatically by pytest. It must NOT import config.py
(which does not exist in Wave 0).
"""
import os
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).parent.parent

CINC_ROOT = PROJECT_ROOT / "data" / "raw" / "cinc2016"
ICBHI_ROOT = PROJECT_ROOT / "data" / "raw" / "icbhi2017"

# ---------------------------------------------------------------------------
# Expected dataset counts (ROADMAP success criteria)
# ---------------------------------------------------------------------------
CINC_EXPECTED = 3126  # total WAVs across training-a through training-e (A-E only, not F)

ICBHI_EXPECTED_WAV = 920   # WAV files in ICBHI 2017
ICBHI_EXPECTED_TXT = 920   # annotation TXT files (one per WAV)

# Per-database recording counts for CinC 2016 (verified: PMC7199391 Table 4)
CINC_DB_COUNTS = {
    "training-a": 409,
    "training-b": 490,
    "training-c": 31,
    "training-d": 55,
    "training-e": 2141,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cinc_available():
    """Return True if the CinC 2016 data directory exists on disk."""
    return CINC_ROOT.exists()


@pytest.fixture
def icbhi_available():
    """Return True if the ICBHI 2017 data directory exists on disk."""
    return ICBHI_ROOT.exists()


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom marks so pytest does not warn about unknown marks."""
    config.addinivalue_line(
        "markers",
        "needs_data: mark test as requiring downloaded dataset(s) — skip if data absent",
    )


# ---------------------------------------------------------------------------
# Synthetic-signal fixture (no data dependency)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_signal():
    """Return a deterministic 5-second mono audio array sampled at 2000 Hz.

    Used by the Phase-2 preprocess tests so they need NO downloaded data: the
    array stands in for a real heart-sound recording when exercising the
    resample / bandpass / segment pure functions in ``src.preprocess`` and
    ``src.segment``.

    The signal is a fixed-seed mix of a 100 Hz sine (in the cardiac band) plus
    low-amplitude Gaussian noise, returned as ``float32``. Returns a dict with
    the array and its native sampling rate so callers do not hard-code the SR.
    """
    import numpy as np  # local import — conftest must not import config at top level

    rng = np.random.default_rng(42)
    fs = 2000           # native rate (PhysioNet/CinC 2016 raw rate)
    duration_s = 5.0
    n = int(duration_s * fs)            # 10000 samples
    t = np.arange(n, dtype=np.float64) / fs
    tone = np.sin(2.0 * np.pi * 100.0 * t)        # 100 Hz cardiac-band tone
    noise = 0.05 * rng.standard_normal(n)
    y = (tone + noise).astype("float32")

    return {"y": y, "sr": fs, "duration_s": duration_s, "n": n}


# ---------------------------------------------------------------------------
# Synthetic-feature-matrix fixture (no data dependency) — Phase 3 Wave 0
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_feature_matrix():
    """Return a deterministic ``(X, y, groups)`` tuple for the Phase-3 train/metric tests.

    No downloaded data is required: a fixed-seed ``numpy.default_rng(42)`` builds a
    small tabular feature matrix that stands in for the cached classical feature
    cache (``features/*_classical.npy``) when exercising the leakage-safe
    ``src.train_classical`` pipelines and the ``src.metrics`` majority-vote helper.

    Shape contract (relied on by the train/metric tests):
      - ``X``: ``(60, 12)`` ``float64`` feature matrix — enough rows/columns to
        fit all four classifiers (logreg / svm / rf / xgb) without warnings.
      - ``y``: length-60 binary label vector (0 = normal, 1 = abnormal) with BOTH
        classes present, so every pipeline can ``.fit`` and ``.predict``.
      - ``groups``: length-60 integer patient-id array of 12 groups × 5 rows each,
        so ``GroupKFold`` / ``StratifiedGroupKFold`` have several disjoint groups
        and the majority-vote test has multiple windows per patient.

    A mild class-dependent shift is added to ``X`` so the toy classifiers are not
    degenerate, while the fixed seed keeps the matrix fully reproducible.
    """
    import numpy as np  # local import — conftest must not import config at top level

    rng = np.random.default_rng(42)
    n_groups = 12
    rows_per_group = 5
    n_rows = n_groups * rows_per_group        # 60
    n_features = 12

    groups = np.repeat(np.arange(n_groups), rows_per_group)   # 12 groups × 5 rows
    # Binary label per group (both classes present), expanded to per-row.
    group_labels = np.array([g % 2 for g in range(n_groups)])  # 0,1,0,1,...
    y = group_labels[groups].astype(int)

    X = rng.standard_normal((n_rows, n_features)).astype("float64")
    # Add a small class-dependent shift so classifiers are separable (non-degenerate).
    X[y == 1] += 0.8

    return X, y, groups
