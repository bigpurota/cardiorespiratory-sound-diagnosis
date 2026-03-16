"""
tests/test_metrics.py — EVAL-01 metric-framework contracts (Phase 3, Wave 0, RED).

Specifies the contracts for the Wave-1 metric functions described in
03-RESEARCH.md §Pattern 5 (heart MAcc = (Se+Sp)/2 at recording level),
§Pattern 6 (lung ICBHI 4-class Score = (Se+Sp)/2 with Se = recall over pooled
abnormal {crackle,wheeze,both}, Sp = recall over normal, normal_label=3) and
§Pattern 7 (degenerate-CM assertion — all predictions in one column raises):

  - src.metrics.heart_macc(y_true_rec, y_pred_rec, y_score_rec=None) -> dict
      with at least {"MAcc", "Se", "Sp"}; toy 2×2 → MAcc 0.75 (Se 0.75 / Sp 0.75).
  - src.metrics.icbhi_score(y_true, y_pred, normal_label=3) -> dict
      with at least {"ICBHI_Score", "Se", "Sp"}; toy 4-class → Se 0.833 / Sp 0.667 / 0.75.
  - src.metrics.majority_vote(window_preds, patient_ids) -> per-patient majority class.
  - src.metrics.assert_not_degenerate(...) raises AssertionError when a confusion
      matrix has all predictions in a single column; passes with ≥2 columns used.

These reference src.metrics symbols that do NOT exist yet, so the tests MUST be
RED now. Collection MUST succeed: imports happen INSIDE each test body
(skip-on-missing), mirroring tests/test_preprocess.py.

Toy values verified live (03-RESEARCH.md §Sources): MAcc=0.750; ICBHI Se=0.833,
Sp=0.667, Score=0.750.
"""
import importlib
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity

# Lung 4-class encoding (03-RESEARCH.md §Pattern 2): normal is the highest index
# so the pooled-abnormal mask is simply ``label != NORMAL_LABEL``.
NORMAL_LABEL = 3   # {crackle:0, wheeze:1, both:2, normal:3}
TOL = 1e-9


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent in Wave 0."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive (Wave 0 module absent)
        pytest.skip(f"{module_name} not implemented yet (Wave 0): {exc}")


# ---------------------------------------------------------------------------
# Heart MAcc — (Se+Sp)/2 on a toy 2×2 case (normal=0, abnormal=1)
# ---------------------------------------------------------------------------

def test_macc_formula():
    """heart_macc returns MAcc == (Se+Sp)/2; toy 2×2 gives Se=0.75, Sp=0.75, MAcc=0.75."""
    metrics = _import("src.metrics")
    if not hasattr(metrics, "heart_macc"):
        pytest.skip("src.metrics.heart_macc not implemented yet (Wave 0)")

    # 4 abnormal (1): 3 correct, 1 wrong  → Se = 3/4 = 0.75
    # 4 normal   (0): 3 correct, 1 wrong  → Sp = 3/4 = 0.75
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])

    out = metrics.heart_macc(y_true, y_pred)
    assert abs(out["Se"] - 0.75) < TOL, f"Se must be 0.75, got {out['Se']}"
    assert abs(out["Sp"] - 0.75) < TOL, f"Sp must be 0.75, got {out['Sp']}"
    assert abs(out["MAcc"] - 0.75) < TOL, f"MAcc must be 0.75, got {out['MAcc']}"
    # MAcc must equal (Se+Sp)/2 by definition.
    assert abs(out["MAcc"] - (out["Se"] + out["Sp"]) / 2) < TOL


# ---------------------------------------------------------------------------
# Lung ICBHI 4-class Score — pooled-abnormal Se / normal Sp
# ---------------------------------------------------------------------------

def test_icbhi_formula():
    """icbhi_score = (Se+Sp)/2 with pooled-abnormal Se / normal Sp; toy → 0.833/0.667/0.75."""
    metrics = _import("src.metrics")
    if not hasattr(metrics, "icbhi_score"):
        pytest.skip("src.metrics.icbhi_score not implemented yet (Wave 0)")

    # 6 abnormal cycles (labels {0,1,2}): 5 predicted as SOME abnormal class, 1 as normal
    #   → Se = 5/6 = 0.8333
    # 3 normal cycles (label 3): 2 predicted normal, 1 predicted abnormal
    #   → Sp = 2/3 = 0.6667
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    y_pred = np.array([0, 1, 1, 0, 2, 3, 3, 3, 0])

    out = metrics.icbhi_score(y_true, y_pred, normal_label=NORMAL_LABEL)
    assert abs(out["Se"] - (5 / 6)) < 1e-6, f"Se must be 0.833, got {out['Se']}"
    assert abs(out["Sp"] - (2 / 3)) < 1e-6, f"Sp must be 0.667, got {out['Sp']}"
    assert abs(out["ICBHI_Score"] - 0.75) < 1e-6, (
        f"ICBHI_Score must be 0.75, got {out['ICBHI_Score']}"
    )
    # Se credits a true-crackle predicted as wheeze (both abnormal) — official metric.
    assert abs(out["ICBHI_Score"] - (out["Se"] + out["Sp"]) / 2) < 1e-6


# ---------------------------------------------------------------------------
# Recording-level majority vote — group windows by patient_id
# ---------------------------------------------------------------------------

def test_majority_vote():
    """majority_vote reduces several windows per patient to the per-patient majority class."""
    metrics = _import("src.metrics")
    if not hasattr(metrics, "majority_vote"):
        pytest.skip("src.metrics.majority_vote not implemented yet (Wave 0)")

    # patient 10: [1,1,0] → majority 1 ; patient 20: [0,0,0,1] → 0 ; patient 30: [1,0] tie-or-1
    patient_ids = np.array([10, 10, 10, 20, 20, 20, 20, 30, 30])
    window_preds = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])

    voted = metrics.majority_vote(window_preds, patient_ids)

    # Accept either a pandas Series indexed by patient id or a (ids, preds) mapping.
    def _lookup(result, pid):
        if hasattr(result, "loc"):           # pandas Series / DataFrame-like
            return int(result.loc[pid])
        if isinstance(result, dict):
            return int(result[pid])
        # array aligned to sorted unique patient ids
        uniq = np.unique(patient_ids)
        idx = int(np.where(uniq == pid)[0][0])
        return int(np.asarray(result)[idx])

    assert _lookup(voted, 10) == 1, "patient 10 majority must be 1"
    assert _lookup(voted, 20) == 0, "patient 20 majority must be 0"
    assert _lookup(voted, 30) == 1, "patient 30 majority must be 1"


# ---------------------------------------------------------------------------
# Degenerate-CM check — all predictions in one column raises AssertionError
# ---------------------------------------------------------------------------

def test_degenerate_check():
    """assert_not_degenerate raises when all preds are one class; passes with ≥2 columns used."""
    metrics = _import("src.metrics")
    if not hasattr(metrics, "assert_not_degenerate"):
        pytest.skip("src.metrics.assert_not_degenerate not implemented yet (Wave 0)")

    labels = [0, 1]

    # Degenerate: every prediction is class 0 → only one column used → must raise.
    y_true_deg = np.array([0, 1, 0, 1, 1])
    y_pred_deg = np.array([0, 0, 0, 0, 0])
    with pytest.raises(AssertionError):
        metrics.assert_not_degenerate(y_true_deg, y_pred_deg, labels=labels)

    # Healthy: both columns used → must NOT raise.
    y_true_ok = np.array([0, 1, 0, 1])
    y_pred_ok = np.array([0, 1, 1, 1])
    metrics.assert_not_degenerate(y_true_ok, y_pred_ok, labels=labels)
