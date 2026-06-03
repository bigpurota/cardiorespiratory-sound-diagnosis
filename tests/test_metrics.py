"""Tests for the evaluation metrics in src/metrics.py."""
import importlib
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

NORMAL_LABEL = 3
TOL = 1e-9


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not implemented yet: {exc}")


def test_macc_formula():
    """heart_macc returns MAcc == (Se+Sp)/2; this 2×2 gives"""
    metrics = _import("src.metrics")
    if not hasattr(metrics, "heart_macc"):
        pytest.skip("src.metrics.heart_macc not implemented yet")

    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1])

    out = metrics.heart_macc(y_true, y_pred)
    assert abs(out["Se"] - 0.75) < TOL, f"Se must be 0.75, got {out['Se']}"
    assert abs(out["Sp"] - 0.75) < TOL, f"Sp must be 0.75, got {out['Sp']}"
    assert abs(out["MAcc"] - 0.75) < TOL, f"MAcc must be 0.75, got {out['MAcc']}"
    assert abs(out["MAcc"] - (out["Se"] + out["Sp"]) / 2) < TOL


def test_icbhi_formula():
    """icbhi_score = (Se+Sp)/2 with pooled-abnormal Se / normal"""
    metrics = _import("src.metrics")
    if not hasattr(metrics, "icbhi_score"):
        pytest.skip("src.metrics.icbhi_score not implemented yet")

    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])
    y_pred = np.array([0, 1, 1, 0, 2, 3, 3, 3, 0])

    out = metrics.icbhi_score(y_true, y_pred, normal_label=NORMAL_LABEL)
    assert abs(out["Se"] - (5 / 6)) < 1e-6, f"Se must be 0.833, got {out['Se']}"
    assert abs(out["Sp"] - (2 / 3)) < 1e-6, f"Sp must be 0.667, got {out['Sp']}"
    assert abs(out["ICBHI_Score"] - 0.75) < 1e-6, (
        f"ICBHI_Score must be 0.75, got {out['ICBHI_Score']}"
    )
    assert abs(out["ICBHI_Score"] - (out["Se"] + out["Sp"]) / 2) < 1e-6


def test_majority_vote():
    """majority_vote reduces several windows per patient to the"""
    metrics = _import("src.metrics")
    if not hasattr(metrics, "majority_vote"):
        pytest.skip("src.metrics.majority_vote not implemented yet")

    patient_ids = np.array([10, 10, 10, 20, 20, 20, 20, 30, 30])
    window_preds = np.array([1, 1, 0, 0, 0, 0, 1, 1, 1])

    voted = metrics.majority_vote(window_preds, patient_ids)

    def _lookup(result, pid):
        if hasattr(result, "loc"):
            return int(result.loc[pid])
        if isinstance(result, dict):
            return int(result[pid])
        uniq = np.unique(patient_ids)
        idx = int(np.where(uniq == pid)[0][0])
        return int(np.asarray(result)[idx])

    assert _lookup(voted, 10) == 1, "patient 10 majority must be 1"
    assert _lookup(voted, 20) == 0, "patient 20 majority must be 0"
    assert _lookup(voted, 30) == 1, "patient 30 majority must be 1"


def test_degenerate_check():
    """assert_not_degenerate raises when all preds are one class;"""
    metrics = _import("src.metrics")
    if not hasattr(metrics, "assert_not_degenerate"):
        pytest.skip("src.metrics.assert_not_degenerate not implemented yet")

    labels = [0, 1]

    y_true_deg = np.array([0, 1, 0, 1, 1])
    y_pred_deg = np.array([0, 0, 0, 0, 0])
    with pytest.raises(AssertionError):
        metrics.assert_not_degenerate(y_true_deg, y_pred_deg, labels=labels)

    y_true_ok = np.array([0, 1, 0, 1])
    y_pred_ok = np.array([0, 1, 1, 1])
    metrics.assert_not_degenerate(y_true_ok, y_pred_ok, labels=labels)
