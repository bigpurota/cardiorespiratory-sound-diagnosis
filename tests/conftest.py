"""Shared fixtures and constants for the cardiorespiratory sound test suite.

Collected automatically by pytest. Deliberately does not import config.py at
module level so collection stays decoupled from the project config.
"""
import os
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

CINC_ROOT = PROJECT_ROOT / "data" / "raw" / "cinc2016"
ICBHI_ROOT = PROJECT_ROOT / "data" / "raw" / "icbhi2017"

CINC_EXPECTED = 3126

ICBHI_EXPECTED_WAV = 920
ICBHI_EXPECTED_TXT = 920

CINC_DB_COUNTS = {
    "training-a": 409,
    "training-b": 490,
    "training-c": 31,
    "training-d": 55,
    "training-e": 2141,
}


@pytest.fixture
def cinc_available():
    """Return True if the CinC 2016 data directory exists on disk."""
    return CINC_ROOT.exists()


@pytest.fixture
def icbhi_available():
    """Return True if the ICBHI 2017 data directory exists on disk."""
    return ICBHI_ROOT.exists()


def pytest_configure(config):
    """Register custom marks so pytest does not warn about unknown marks."""
    config.addinivalue_line(
        "markers",
        "needs_data: mark test as requiring downloaded dataset(s) — skip if data absent",
    )


@pytest.fixture
def synthetic_signal():
    """A deterministic 5-second mono audio array at 2000 Hz (no data needed).

    Stands in for a real heart-sound recording when exercising the resample /
    bandpass / segment functions in ``src.preprocess`` and ``src.segment``.

    The signal is a fixed-seed mix of a 100 Hz sine (in the cardiac band) plus
    low-amplitude Gaussian noise, as ``float32``. Returns a dict with the array
    and its native sampling rate so callers do not hard-code the SR.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    fs = 2000
    duration_s = 5.0
    n = int(duration_s * fs)
    t = np.arange(n, dtype=np.float64) / fs
    tone = np.sin(2.0 * np.pi * 100.0 * t)
    noise = 0.05 * rng.standard_normal(n)
    y = (tone + noise).astype("float32")

    return {"y": y, "sr": fs, "duration_s": duration_s, "n": n}


@pytest.fixture
def synthetic_feature_matrix():
    """A deterministic ``(X, y, groups)`` tuple for the train/metric tests.

    A fixed-seed ``numpy.default_rng(42)`` builds a small tabular feature matrix
    that stands in for the cached classical features (``features/*_classical.npy``)
    when exercising the leakage-safe ``src.train_classical`` pipelines and the
    ``src.metrics`` majority-vote helper.

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
    import numpy as np

    rng = np.random.default_rng(42)
    n_groups = 12
    rows_per_group = 5
    n_rows = n_groups * rows_per_group
    n_features = 12

    groups = np.repeat(np.arange(n_groups), rows_per_group)
    group_labels = np.array([g % 2 for g in range(n_groups)])
    y = group_labels[groups].astype(int)

    X = rng.standard_normal((n_rows, n_features)).astype("float64")
    X[y == 1] += 0.8

    return X, y, groups


@pytest.fixture
def synthetic_spectrogram_cache():
    """A deterministic 2-class heart + 4-class lung log-mel cache (no audio).

    A fixed-seed ``numpy.default_rng(42)`` builds two small ``(N, 64, 128)`` float32
    log-mel "cache" payloads that stand in for the spectrogram cache
    (``features/*_logmel.npy``) when exercising the leakage-safe ``src.datasets`` /
    ``src.cnn`` / ``src.train_cnn`` pipelines and the ``src.metrics`` suite. The smoke
    tests train a tiny model on this fixture so they need no real PhysioNet/ICBHI WAVs.

    Returns a dict ``{"heart": <payload>, "lung": <payload>}`` where each payload is a
    dict mirroring the on-disk spectrogram-cache schema:

      - ``X``: ``(N, 64, 128)`` ``float32`` log-mel-spectrogram tensor stack.
      - ``labels``: length-N int label vector — heart in ``{0, 1}`` (HEART_LABEL_MAP
        codomain), lung in ``{0, 1, 2, 3}`` (LUNG_LABEL_MAP codomain). ALL classes
        are present so every Dataset/loader can ``.fit``/``.predict`` and the weighted
        loss + non-degenerate-CM guards hold.
      - ``patient_id``: length-N ``str`` patient/group ids — ``>=4`` distinct groups so
        a ``GroupShuffleSplit`` validation carve has disjoint train/val groups.
      - ``split``: length-N ``str`` in ``{"train", "test"}`` — BOTH present (patient-
        level, so a patient never spans train+test), to exercise the train-only class
        weights / augment-after-split contract.
      - ``recording_id``: length-N ``str`` — heart ``recording_id == patient_id`` (one
        recording per patient), lung a WAV-stem-style id (several cycles per recording).

    A small class-dependent shift is added to ``X`` so a smoke-trained model is
    non-degenerate (uses >=2 predicted CM columns), while the fixed seed keeps the
    cache fully reproducible. ``N`` is kept small (heart 40, lung 64 rows) so smoke
    training runs in seconds on CPU.
    """
    import numpy as np

    rng = np.random.default_rng(42)

    def _build(n_classes, n_per_class, groups_per_class, id_prefix, heart_like):
        """Build one cache payload with all classes present and patient-level splits."""
        H, W = 64, 128
        X_list, labels, patient_id, split, recording_id = [], [], [], [], []
        for cls in range(n_classes):
            for g in range(groups_per_class):
                pid = f"{id_prefix}{cls}_{g:02d}"
                this_split = "train" if (g % 2 == 0) else "test"
                rows = n_per_class // groups_per_class
                for r in range(rows):
                    spec = rng.standard_normal((H, W)).astype("float32")
                    spec += float(cls) * 0.8
                    X_list.append(spec)
                    labels.append(cls)
                    patient_id.append(pid)
                    split.append(this_split)
                    recording_id.append(pid if heart_like else f"{pid}.wav")
        X = np.stack(X_list).astype("float32")
        return {
            "X": X,
            "labels": np.asarray(labels, dtype=int),
            "patient_id": np.asarray(patient_id, dtype=object),
            "split": np.asarray(split, dtype=object),
            "recording_id": np.asarray(recording_id, dtype=object),
        }

    heart = _build(n_classes=2, n_per_class=20, groups_per_class=4,
                   id_prefix="h", heart_like=True)
    lung = _build(n_classes=4, n_per_class=16, groups_per_class=4,
                  id_prefix="l", heart_like=False)

    return {"heart": heart, "lung": lung}
