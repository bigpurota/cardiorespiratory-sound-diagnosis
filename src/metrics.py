"""
src/metrics.py — evaluation metric framework (Phase 3, EVAL-01).

Stateless, toy-input-testable metrics shared by every classical (and later CNN)
experiment so the metric definition is identical across models (PITFALLS Pitfall 3):

  - ``majority_vote(window_preds, patient_ids)`` — reduce per-window predictions to
    one per recording (group by patient_id → majority class). Returns a pandas Series
    indexed by patient_id. (03-RESEARCH.md §Pattern 5.)
  - ``heart_macc(y_true_rec, y_pred_rec, y_score_rec=None)`` — recording-level
    {MAcc=(Se+Sp)/2, Se=recall(abnormal=1), Sp=recall(normal=0), macro_f1, accuracy,
    + auc_roc when a recording-level score is supplied}. (D-08, §Pattern 5.)
  - ``icbhi_score(y_true, y_pred, normal_label=3)`` — cycle-level ICBHI 4-class Score:
    Se = recall over POOLED abnormal ({crackle,wheeze,both}, i.e. ``label != normal_label``),
    Sp = recall over normal, ICBHI_Score = (Se+Sp)/2. The official metric credits a
    true-crackle predicted as wheeze (both abnormal) — it measures normal-vs-abnormal
    discrimination, not 4-way accuracy. (D-09, §Pattern 6.)
  - ``assert_not_degenerate(y_true, y_pred, labels)`` — raise AssertionError when a
    classifier emits predictions in fewer than 2 confusion-matrix columns ("predict-one-
    class" degeneracy). Returns the column count otherwise. (D-10, §Pattern 7.)
  - ``save_cm(y_true, y_pred, labels, title, out_png)`` — headless (Agg) confusion-matrix
    figure; calls ``assert_not_degenerate`` first. (§Pattern 7.)
  - Helpers: ``per_class_se``, ``macro_f1``, ``accuracy``.

``import config`` runs first (SEED=42 side effect). The matplotlib Agg backend is set
BEFORE importing pyplot so figures render with no display (safe under ``uv run``).
"""
import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # MUST precede pyplot import — headless (§Pattern 7)
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
    ConfusionMatrixDisplay,
)

__all__ = [
    "majority_vote",
    "heart_macc",
    "icbhi_score",
    "assert_not_degenerate",
    "save_cm",
    "per_class_se",
    "macro_f1",
    "accuracy",
]


# ---------------------------------------------------------------------------
# §Pattern 5 — recording-level majority vote
# ---------------------------------------------------------------------------
def majority_vote(window_preds, patient_ids):
    """Reduce per-window predictions to one per recording via per-patient majority class.

    Returns a pandas Series indexed by patient_id. Ties resolve to the smaller index of
    the tied classes (pandas ``value_counts().idxmax()`` — deterministic).
    """
    df = pd.DataFrame({"pid": np.asarray(patient_ids), "pred": np.asarray(window_preds)})
    return df.groupby("pid")["pred"].agg(lambda s: s.value_counts().idxmax())


# ---------------------------------------------------------------------------
# §Pattern 5 — heart MAcc (recording-level)
# ---------------------------------------------------------------------------
def heart_macc(y_true_rec, y_pred_rec, y_score_rec=None):
    """Recording-level heart metric dict (MAcc, Se, Sp, macro_f1, accuracy[, auc_roc]).

    Labels: normal=0, abnormal=1 (map the manifest -1/1 convention BEFORE calling).
    Se = recall(abnormal), Sp = recall(normal), MAcc = (Se+Sp)/2 — the official CinC
    2016 metric. ``y_score_rec`` (mean abnormal-class probability per recording) adds AUC.
    """
    y_true_rec = np.asarray(y_true_rec)
    y_pred_rec = np.asarray(y_pred_rec)
    tn, fp, fn, tp = confusion_matrix(y_true_rec, y_pred_rec, labels=[0, 1]).ravel()
    Se = tp / (tp + fn) if (tp + fn) else 0.0   # recall abnormal
    Sp = tn / (tn + fp) if (tn + fp) else 0.0   # recall normal
    out = {
        "MAcc": (Se + Sp) / 2,
        "Se": Se,
        "Sp": Sp,
        "macro_f1": f1_score(y_true_rec, y_pred_rec, average="macro"),
        "accuracy": float((y_true_rec == y_pred_rec).mean()),
    }
    if y_score_rec is not None:
        # AUC needs a recording-level score: mean abnormal-class probability per recording
        # (§Pitfall 5). Guard the single-class edge case where roc_auc_score is undefined.
        try:
            out["auc_roc"] = roc_auc_score(y_true_rec, y_score_rec)
        except ValueError:
            out["auc_roc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# §Pattern 6 — lung ICBHI 4-class Score (cycle-level, pooled-abnormal Se / normal Sp)
# ---------------------------------------------------------------------------
def icbhi_score(y_true, y_pred, normal_label=3):
    """Official ICBHI 4-class Score: (Se+Sp)/2 with pooled-abnormal Se / normal Sp.

    Se = fraction of true-abnormal cycles ({crackle,wheeze,both} == ``label != normal_label``)
    predicted as ANY abnormal class; Sp = fraction of true-normal cycles predicted normal.
    Se therefore credits a true-crackle predicted as wheeze (both abnormal) — it measures
    normal-vs-abnormal discrimination, the official metric (report per-class Se alongside).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    is_ab_t = y_true != normal_label
    is_ab_p = y_pred != normal_label
    Se = is_ab_p[is_ab_t].sum() / is_ab_t.sum() if is_ab_t.sum() else 0.0
    Sp = (y_pred[~is_ab_t] == normal_label).sum() / (~is_ab_t).sum() if (~is_ab_t).sum() else 0.0
    return {"ICBHI_Score": (Se + Sp) / 2, "Se": float(Se), "Sp": float(Sp)}


# ---------------------------------------------------------------------------
# Secondary helpers (per-class Se, macro-F1, accuracy)
# ---------------------------------------------------------------------------
def per_class_se(y_true, y_pred, labels):
    """Per-class recall (sensitivity) as a {label: recall} dict (D-09)."""
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {lab: float(r) for lab, r in zip(labels, rec)}


def macro_f1(y_true, y_pred):
    """Macro-averaged F1 (unweighted mean over classes)."""
    return float(f1_score(y_true, y_pred, average="macro"))


def accuracy(y_true, y_pred):
    """Plain accuracy (secondary; never the headline metric on imbalanced data)."""
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


# ---------------------------------------------------------------------------
# §Pattern 7 — degenerate-classifier guard + headless CM figure (D-10)
# ---------------------------------------------------------------------------
def assert_not_degenerate(y_true, y_pred, labels):
    """Raise AssertionError when predictions occupy fewer than 2 confusion columns.

    A "predict-one-class-always" model (e.g. always-normal on imbalanced lung) uses a
    single predicted-class column. Returns the number of columns used otherwise (D-10).
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cols_used = int((cm.sum(axis=0) > 0).sum())
    assert cols_used >= 2, (
        f"DEGENERATE: all predictions in {cols_used} column(s) of {len(labels)} classes"
    )
    return cols_used


def save_cm(y_true, y_pred, labels, title, out_png):
    """Render and save a confusion-matrix PNG (Agg backend); guards degeneracy first."""
    cols_used = assert_not_degenerate(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False)
    # Smaller title + tight bbox so long internal titles are not clipped at the
    # figure edge (the saved canvas expands to contain the full title text).
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cols_used
