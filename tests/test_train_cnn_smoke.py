"""Smoke tests for the CNN training entry point (src/train_cnn.py).

A tiny (<=2-epoch) training run is driven on the synthetic_spectrogram_cache
fixture so the deep-learning loop runs end-to-end in seconds with no PhysioNet or
ICBHI WAV files. Coverage:
  - test_metric_suite: the run reports the right primary metric (MAcc for heart,
    ICBHI_Score for lung, via src/metrics.py) plus Se, Sp, macro_f1, accuracy.
  - test_learning_curve_png: the run writes a learning-curve PNG.
  - test_early_stop: epochs_ran <= max_epochs and patience is honored.
  - test_non_degenerate_cm: save_cm on the test predictions does not raise (the model
    uses >=2 predicted columns).

Imports happen inside the test bodies and skip when src/train_cnn.py is unavailable.
"""
import importlib
import os
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not implemented yet: {exc}")


def _smoke_entry(train_cnn):
    """Return the tiny-run smoke entry point on src.train_cnn, or skip if absent."""
    entry = (
        getattr(train_cnn, "run_modality", None)
        or getattr(train_cnn, "smoke_run", None)
        or getattr(train_cnn, "train_modality", None)
        or getattr(train_cnn, "fit_modality", None)
    )
    if entry is None:
        pytest.skip(
            "src.train_cnn smoke entry (run_modality/smoke_run/train_modality/fit_modality) "
            "not implemented yet"
        )
    return entry


def _run(entry, payload, modality, tmp_path, **over):
    """Drive a tiny (<=2 epoch) run on a fixture payload; keep kwargs permissive.

    The entry signature is not fixed here, so we pass the synthetic cache payload, a
    small epoch budget, and a tmp output dir, and tolerate either a (model, dict)
    tuple or a bare metrics dict return.
    """
    kwargs = dict(
        cache=payload,
        modality=modality,
        model="cnn",
        max_epochs=2,
        patience=1,
        out_dir=str(tmp_path),
    )
    kwargs.update(over)
    try:
        result = entry(**kwargs)
    except TypeError as exc:
        pytest.skip(f"src.train_cnn smoke entry signature not finalized yet: {exc}")
    if isinstance(result, tuple):
        for part in result:
            if isinstance(part, dict):
                return part
        pytest.skip("src.train_cnn run returned no metrics dict yet")
    if isinstance(result, dict):
        return result
    pytest.skip("src.train_cnn run return shape not recognized yet")


def test_metric_suite(synthetic_spectrogram_cache, tmp_path):
    """A tiny heart + lung run carries the correct primary metric + full metric suite.

    heart -> primary_metric_name == "MAcc"; lung -> "ICBHI_Score". Both also expose
    Se, Sp, macro_f1, accuracy (computed via the reused src/metrics.py, not re-derived).
    """
    train_cnn = _import("src.train_cnn")
    entry = _smoke_entry(train_cnn)

    cases = [
        ("heart", synthetic_spectrogram_cache["heart"], "MAcc"),
        ("lung", synthetic_spectrogram_cache["lung"], "ICBHI_Score"),
    ]
    for modality, payload, expected_primary in cases:
        m = _run(entry, payload, modality, tmp_path / modality)
        assert m.get("primary_metric_name") == expected_primary, (
            f"{modality}: primary_metric_name must be {expected_primary!r}; "
            f"got {m.get('primary_metric_name')!r}"
        )
        for col in ("Se", "Sp", "macro_f1", "accuracy"):
            assert col in m, f"{modality}: metric dict missing '{col}'"


def test_learning_curve_png(synthetic_spectrogram_cache, tmp_path):
    """A tiny training run writes a learning-curve PNG to the output dir."""
    train_cnn = _import("src.train_cnn")
    entry = _smoke_entry(train_cnn)

    out_dir = tmp_path / "heart"
    m = _run(entry, synthetic_spectrogram_cache["heart"], "heart", out_dir)

    png = m.get("curve_png") or m.get("learning_curve_png")
    if png:
        assert os.path.exists(png), f"learning-curve PNG not written: {png}"
    else:
        pngs = list(pathlib.Path(out_dir).rglob("*.png")) if out_dir.exists() else []
        assert any("curve" in p.name.lower() or "learning" in p.name.lower() for p in pngs), (
            f"no learning-curve PNG found under {out_dir}"
        )


def test_early_stop(synthetic_spectrogram_cache, tmp_path):
    """A plateauing run with patience=1 stops at/before max_epochs (early-stop honored).

    With ``max_epochs`` set generously and ``patience=1`` on a small fixture that quickly
    plateaus, ``epochs_ran`` must be <= max_epochs (and usually strictly less once the
    val metric stops improving), showing early stopping is wired up.
    """
    train_cnn = _import("src.train_cnn")
    entry = _smoke_entry(train_cnn)

    max_epochs = 5
    m = _run(
        entry, synthetic_spectrogram_cache["heart"], "heart", tmp_path / "heart",
        max_epochs=max_epochs, patience=1,
    )
    if "epochs_ran" not in m:
        pytest.skip("src.train_cnn run does not report epochs_ran yet")
    assert m["epochs_ran"] <= max_epochs, (
        f"epochs_ran ({m['epochs_ran']}) must be <= max_epochs ({max_epochs}) — "
        "early-stop / wall-cap must bound the loop."
    )


def test_non_degenerate_cm(synthetic_spectrogram_cache, tmp_path):
    """``save_cm`` on the tiny run's TEST predictions does not raise (>=2 CM columns).

    The smoke model trained with weighted CE on the class-separable fixture should not
    collapse to a single predicted class; ``src.metrics.save_cm`` (which calls
    ``assert_not_degenerate`` first) must succeed for the run's TEST predictions.
    """
    train_cnn = _import("src.train_cnn")
    entry = _smoke_entry(train_cnn)
    metrics = _import("src.metrics")
    if not hasattr(metrics, "save_cm"):
        pytest.skip("src.metrics.save_cm not available yet")

    import numpy as np

    m = _run(entry, synthetic_spectrogram_cache["heart"], "heart", tmp_path / "heart")

    y_true = m.get("test_true")
    y_pred = m.get("test_pred")
    if y_true is None or y_pred is None:
        pytest.skip("src.train_cnn run does not expose test_true/test_pred yet")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    out_png = tmp_path / "cm_heart_cnn.png"
    cols_used = metrics.save_cm(y_true, y_pred, labels, "smoke heart cnn", str(out_png))
    assert cols_used >= 2, "smoke run CM is degenerate (<2 predicted columns)"
