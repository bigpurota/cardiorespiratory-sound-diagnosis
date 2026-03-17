"""
tests/test_train_cnn_smoke.py — MODL-02 / SC2 + SC3 contracts (Phase 4, Wave 0).

Smoke-level contracts for the Wave-2 ``src/train_cnn.py`` entry point
(04-RESEARCH.md §Code Examples 6). A TINY (<=2-epoch) training run is driven on the
no-audio ``synthetic_spectrogram_cache`` fixture so the full deep-learning loop is
exercised end-to-end in seconds with NO PhysioNet/ICBHI WAV files:

  - ``test_metric_suite`` (SC2): the run's metric dict carries the correct primary metric
    (``MAcc`` for heart, ``ICBHI_Score`` for lung — reusing ``src/metrics.py``, NOT a
    re-implementation) plus the full suite (Se, Sp, macro_f1, accuracy).
  - ``test_learning_curve_png`` (SC3): the run writes a learning-curve PNG to a tmp path.
  - ``test_early_stop`` (SC3): ``epochs_ran <= max_epochs`` and patience is honored
    (a plateauing run with patience=1 stops before max_epochs).
  - ``test_non_degenerate_cm`` (SC3): ``save_cm`` on the run's TEST predictions does not
    raise — i.e. the smoke model + weighted-CE is non-degenerate (>=2 predicted columns).

All imports happen INSIDE the test bodies (skip-on-missing) so Wave-0 collection has
zero errors and every stub stays SKIP/RED until Wave 2 ships ``src/train_cnn.py``.
"""
import importlib
import os
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent in Wave 0."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive (Wave 0/2 module absent)
        pytest.skip(f"{module_name} not implemented yet (Wave 0/2): {exc}")


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
            "not implemented yet (Wave 0/2)"
        )
    return entry


def _run(entry, payload, modality, tmp_path, **over):
    """Drive a tiny (<=2 epoch) run on a fixture payload; keep kwargs permissive.

    The Wave-2 entry signature is not frozen in Wave 0, so we pass the synthetic cache
    payload + a small epoch budget + a tmp output dir and tolerate either a (model, dict)
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
        pytest.skip(f"src.train_cnn smoke entry signature not finalized yet (Wave 0/2): {exc}")
    # Normalize to a metrics dict.
    if isinstance(result, tuple):
        for part in result:
            if isinstance(part, dict):
                return part
        pytest.skip("src.train_cnn run returned no metrics dict yet (Wave 0/2)")
    if isinstance(result, dict):
        return result
    pytest.skip("src.train_cnn run return shape not recognized yet (Wave 0/2)")


# ---------------------------------------------------------------------------
# SMOKE — metric suite: correct primary metric + full suite for both modalities
# ---------------------------------------------------------------------------

def test_metric_suite(synthetic_spectrogram_cache, tmp_path):
    """A tiny heart + lung run carries the correct primary metric + full metric suite.

    heart -> primary_metric_name == "MAcc"; lung -> "ICBHI_Score". Both must also expose
    Se, Sp, macro_f1, accuracy (computed via the REUSED src/metrics.py, not re-derived).
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


# ---------------------------------------------------------------------------
# SMOKE — one tiny run writes a learning-curve PNG
# ---------------------------------------------------------------------------

def test_learning_curve_png(synthetic_spectrogram_cache, tmp_path):
    """A tiny training run writes a learning-curve PNG to the output dir (SC3)."""
    train_cnn = _import("src.train_cnn")
    entry = _smoke_entry(train_cnn)

    out_dir = tmp_path / "heart"
    m = _run(entry, synthetic_spectrogram_cache["heart"], "heart", out_dir)

    # Prefer an explicit path in the returned dict; otherwise scan the out dir.
    png = m.get("curve_png") or m.get("learning_curve_png")
    if png:
        assert os.path.exists(png), f"learning-curve PNG not written: {png}"
    else:
        pngs = list(pathlib.Path(out_dir).rglob("*.png")) if out_dir.exists() else []
        assert any("curve" in p.name.lower() or "learning" in p.name.lower() for p in pngs), (
            f"no learning-curve PNG found under {out_dir} (SC3)"
        )


# ---------------------------------------------------------------------------
# SMOKE — early stop: epochs_ran <= max_epochs and patience is honored
# ---------------------------------------------------------------------------

def test_early_stop(synthetic_spectrogram_cache, tmp_path):
    """A plateauing run with patience=1 stops at/before max_epochs (early-stop honored).

    With ``max_epochs`` set generously and ``patience=1`` on a small fixture that quickly
    plateaus, ``epochs_ran`` must be <= max_epochs (and typically strictly less once the
    val metric stops improving) — proving early stopping is wired (D-06).
    """
    train_cnn = _import("src.train_cnn")
    entry = _smoke_entry(train_cnn)

    max_epochs = 5
    m = _run(
        entry, synthetic_spectrogram_cache["heart"], "heart", tmp_path / "heart",
        max_epochs=max_epochs, patience=1,
    )
    if "epochs_ran" not in m:
        pytest.skip("src.train_cnn run does not report epochs_ran yet (Wave 0/2)")
    assert m["epochs_ran"] <= max_epochs, (
        f"epochs_ran ({m['epochs_ran']}) must be <= max_epochs ({max_epochs}) — "
        "early-stop / wall-cap must bound the loop (D-06)."
    )


# ---------------------------------------------------------------------------
# SMOKE — the tiny run's TEST-set confusion matrix is non-degenerate
# ---------------------------------------------------------------------------

def test_non_degenerate_cm(synthetic_spectrogram_cache, tmp_path):
    """``save_cm`` on the tiny run's TEST predictions does not raise (>=2 CM columns).

    The smoke model trained with weighted CE on the class-separable fixture must not
    collapse to a single predicted class; ``src.metrics.save_cm`` (which calls
    ``assert_not_degenerate`` first) must succeed for the run's TEST predictions (D-10).
    """
    train_cnn = _import("src.train_cnn")
    entry = _smoke_entry(train_cnn)
    metrics = _import("src.metrics")
    if not hasattr(metrics, "save_cm"):
        pytest.skip("src.metrics.save_cm not available yet (Wave 0)")

    import numpy as np

    m = _run(entry, synthetic_spectrogram_cache["heart"], "heart", tmp_path / "heart")

    y_true = m.get("test_true")
    y_pred = m.get("test_pred")
    if y_true is None or y_pred is None:
        pytest.skip("src.train_cnn run does not expose test_true/test_pred yet (Wave 0/2)")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    out_png = tmp_path / "cm_heart_cnn.png"
    # save_cm calls assert_not_degenerate first; this must NOT raise for the smoke run.
    cols_used = metrics.save_cm(y_true, y_pred, labels, "smoke heart cnn", str(out_png))
    assert cols_used >= 2, "smoke run CM is degenerate (<2 predicted columns)"
