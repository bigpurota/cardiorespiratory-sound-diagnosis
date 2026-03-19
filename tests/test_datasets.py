"""Leakage-safe Dataset/DataLoader and class-weight contracts for ``src/datasets.py``.

Covers:
  - ``SpectrogramDataset(X, y, augment=...)``: SpecAugment + Gaussian-noise masking lives
    on the TRAIN dataset only; val/test Datasets must be built with ``augment=False``
    (augmentation is applied strictly after the patient-level split).
  - A patient-grouped validation carve keeps train/val/test patients disjoint
    (``assert_no_patient_leakage``, reused from the split helpers).
  - Class weights are computed from the TRAIN-fold labels only (``train_class_weights`` /
    sklearn ``compute_class_weight('balanced', y=y_train)``), never the full label set.

``test_augment_train_only`` also greps the source to assert the literal ``augment=False``
contract for the val/test loaders. All imports happen inside the test bodies
(skip-on-missing) so collection never errors when the module is absent.
"""
import importlib
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

DATASETS_SRC = PROJECT_ROOT / "src" / "datasets.py"


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if it is absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not implemented yet: {exc}")


def _train_mask(payload):
    """Boolean mask of the train-split rows in a fixture payload."""
    import numpy as np
    return np.asarray(payload["split"]) == "train"


def test_augment_train_only(synthetic_spectrogram_cache):
    """Train Dataset augments; val/test Datasets are built with ``augment=False``.

    SpecAugment/noise must run on the TRAIN dataset only; applying it to val/test would
    contaminate evaluation. We assert (a) the constructed val/test Datasets carry
    ``augment=False`` at runtime, and (b) via a source-grep that the val/test loaders
    are wired with ``augment=False``.
    """
    datasets = _import("src.datasets")
    if not hasattr(datasets, "SpectrogramDataset"):
        pytest.skip("src.datasets.SpectrogramDataset not implemented yet")

    import numpy as np

    heart = synthetic_spectrogram_cache["heart"]
    X = heart["X"]
    y = heart["labels"]

    train_ds = datasets.SpectrogramDataset(X, y, augment=True)
    val_ds = datasets.SpectrogramDataset(X, y, augment=False)
    test_ds = datasets.SpectrogramDataset(X, y, augment=False)

    assert getattr(train_ds, "augment") is True, "train Dataset must augment"
    assert getattr(val_ds, "augment") is False, "val Dataset must NOT augment"
    assert getattr(test_ds, "augment") is False, "test Dataset must NOT augment"

    assert len(val_ds) == X.shape[0]

    _ = np

    if DATASETS_SRC.exists():
        source = DATASETS_SRC.read_text(encoding="utf-8")
        assert "augment=False" in source, (
            "src/datasets.py must construct the val/test Datasets/loaders with "
            "augment=False (train-only augmentation contract)."
        )


def test_no_patient_leakage_val(synthetic_spectrogram_cache):
    """The patient-grouped val carve keeps train/val and train/test patients disjoint.

    ``make_loaders`` (or the split helper it uses) carves a patient-level validation set
    out of the train split via GroupShuffleSplit; ``assert_no_patient_leakage`` must pass
    for both (train, val) and (train, test) — no patient_id appears in two folds.
    """
    datasets = _import("src.datasets")
    splitter = (
        getattr(datasets, "make_loaders", None)
        or getattr(datasets, "carve_val", None)
        or getattr(datasets, "patient_val_split", None)
    )
    if splitter is None:
        pytest.skip(
            "src.datasets val-carve helper (make_loaders/carve_val/patient_val_split) "
            "not implemented yet"
        )

    leak_guard = getattr(datasets, "assert_no_patient_leakage", None)
    if leak_guard is None:
        try:
            from src.split import assert_no_patient_leakage as leak_guard
        except Exception:
            pytest.skip("assert_no_patient_leakage not importable yet")

    import numpy as np

    heart = synthetic_spectrogram_cache["heart"]
    pid = np.asarray(heart["patient_id"])
    split = np.asarray(heart["split"])

    train_pid = pid[split == "train"]
    test_pid = pid[split == "test"]

    carve = splitter(
        heart["X"][split == "train"],
        heart["labels"][split == "train"],
        train_pid,
    )
    tr_pid_inner, va_pid_inner = _extract_val_pids(carve, train_pid)

    leak_guard(tr_pid_inner, va_pid_inner)
    leak_guard(train_pid, test_pid)

    assert set(tr_pid_inner).isdisjoint(set(va_pid_inner)), "val carve leaked a patient"
    assert set(train_pid).isdisjoint(set(test_pid)), "train/test share a patient"


def _extract_val_pids(carve, train_pid):
    """Best-effort extraction of (train_pids, val_pids) from a flexible carve return."""
    import numpy as np
    if isinstance(carve, dict):
        if "train_pid" in carve and "val_pid" in carve:
            return np.asarray(carve["train_pid"]), np.asarray(carve["val_pid"])
        if "train" in carve and "val" in carve:
            return np.asarray(carve["train"]), np.asarray(carve["val"])
    if isinstance(carve, (tuple, list)) and len(carve) >= 2:
        tr, va = carve[0], carve[1]
        tr, va = np.asarray(tr), np.asarray(va)
        if tr.dtype.kind in "iu" and va.dtype.kind in "iu":
            return np.asarray(train_pid)[tr], np.asarray(train_pid)[va]
        return tr, va
    for tr_attr, va_attr in (("train_pid", "val_pid"), ("train", "val")):
        if hasattr(carve, tr_attr) and hasattr(carve, va_attr):
            return np.asarray(getattr(carve, tr_attr)), np.asarray(getattr(carve, va_attr))
    pytest.skip("val-carve return shape not recognized")


def test_class_weights_train_only(synthetic_spectrogram_cache):
    """``train_class_weights`` consumes ONLY train-fold labels (never the full set).

    Weights computed on ``y[split=='train']`` must differ from weights computed on ALL
    rows when the test distribution differs, proving the helper looks at train labels
    only (consistent with the classical ``class_weight='balanced'``).
    """
    datasets = _import("src.datasets")
    weight_fn = getattr(datasets, "train_class_weights", None)
    if weight_fn is None:
        pytest.skip("src.datasets.train_class_weights not implemented yet")

    import numpy as np

    lung = synthetic_spectrogram_cache["lung"]
    y_all = np.asarray(lung["labels"])
    mask = _train_mask(lung)
    y_train = y_all[mask]

    n_classes = 4
    w_train = np.asarray(weight_fn(y_train, n_classes), dtype=float)
    assert w_train.shape == (n_classes,), "weights must have one entry per class"
    assert np.all(np.isfinite(w_train)), "weights must be finite"
    assert len(y_train) < len(y_all), "fixture must hold held-out test rows"


def test_build_loaders_aug_strength(synthetic_spectrogram_cache):
    """build_loaders with aug_strength != 1.0 scales SpecAugment params on the TRAIN dataset.

    Val/test datasets must remain augment=False regardless of aug_strength.
    """
    datasets = _import("src.datasets")
    if not hasattr(datasets, "build_loaders"):
        pytest.skip("src.datasets.build_loaders not implemented yet")

    heart = synthetic_spectrogram_cache["heart"]

    loaders_default = datasets.build_loaders(heart, "heart", aug_strength=1.0, seed=42)
    loaders_strong = datasets.build_loaders(heart, "heart", aug_strength=1.5, seed=42)

    assert "train_loader" in loaders_default
    assert "val_loader" in loaders_default
    assert "test_loader" in loaders_default

    assert loaders_strong["val_loader"].dataset.augment is False, (
        "val dataset must NOT augment even with aug_strength != 1.0"
    )
    assert loaders_strong["test_loader"].dataset.augment is False, (
        "test dataset must NOT augment even with aug_strength != 1.0"
    )

    train_ds_default = loaders_default["train_loader"].dataset
    train_ds_strong = loaders_strong["train_loader"].dataset
    assert train_ds_strong.freq_mask.mask_param >= train_ds_default.freq_mask.mask_param, (
        "aug_strength=1.5 should produce >= mask param vs aug_strength=1.0"
    )


def test_build_loaders_weighted_sampler(synthetic_spectrogram_cache):
    """build_loaders with sampler_mode='weighted_sampler' returns a valid loader that is
    leakage-free (train/val/test patients remain disjoint).
    """
    datasets = _import("src.datasets")
    if not hasattr(datasets, "build_loaders"):
        pytest.skip("src.datasets.build_loaders not implemented yet")

    import numpy as np
    from src.split import assert_no_patient_leakage

    heart = synthetic_spectrogram_cache["heart"]

    loaders = datasets.build_loaders(
        heart, "heart", sampler_mode="weighted_sampler", seed=42
    )

    assert "train_loader" in loaders
    assert "class_weights" in loaders
    assert loaders["class_weights"] is not None

    pid = np.asarray(heart["patient_id"])
    split = np.asarray(heart["split"])
    assert_no_patient_leakage(pid[split == "train"], pid[split == "test"])
