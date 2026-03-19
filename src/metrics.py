"""
Evaluation metrics shared by every experiment so definitions stay identical across models.

Provides recording-level majority voting, the CinC 2016 heart MAcc, the official ICBHI
4-class Score, a degenerate-classifier guard, a confusion-matrix figure helper, and a few
small per-class/F1/accuracy helpers. The matplotlib Agg backend is set before importing
pyplot so figures render with no display.
"""
from src import config

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
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


def majority_vote(window_preds, patient_ids):
    """Reduce per-window predictions to one per recording via per-patient majority class.

    Returns a pandas Series indexed by patient_id. Ties resolve deterministically to the
    smaller class index (``value_counts().idxmax()``).
    """
    df = pd.DataFrame({"pid": np.asarray(patient_ids), "pred": np.asarray(window_preds)})
    return df.groupby("pid")["pred"].agg(lambda s: s.value_counts().idxmax())


def heart_macc(y_true_rec, y_pred_rec, y_score_rec=None):
    """Recording-level heart metric dict (MAcc, Se, Sp, macro_f1, accuracy[, auc_roc]).

    Labels: normal=0, abnormal=1 (map the manifest -1/1 convention before calling).
    Se = recall(abnormal), Sp = recall(normal), MAcc = (Se+Sp)/2 (the official CinC 2016
    metric). Passing ``y_score_rec`` (mean abnormal-class probability per recording) adds AUC.
    """
    y_true_rec = np.asarray(y_true_rec)
    y_pred_rec = np.asarray(y_pred_rec)
    tn, fp, fn, tp = confusion_matrix(y_true_rec, y_pred_rec, labels=[0, 1]).ravel()
    Se = tp / (tp + fn) if (tp + fn) else 0.0
    Sp = tn / (tn + fp) if (tn + fp) else 0.0
    out = {
        "MAcc": (Se + Sp) / 2,
        "Se": Se,
        "Sp": Sp,
        "macro_f1": f1_score(y_true_rec, y_pred_rec, average="macro"),
        "accuracy": float((y_true_rec == y_pred_rec).mean()),
    }
    if y_score_rec is not None:
        try:
            out["auc_roc"] = roc_auc_score(y_true_rec, y_score_rec)
        except ValueError:
            out["auc_roc"] = float("nan")
    return out


def icbhi_score(y_true, y_pred, normal_label=3):
    """Official ICBHI 4-class Score: (Se+Sp)/2 with pooled-abnormal Se and normal Sp.

    Se = fraction of true-abnormal cycles (``label != normal_label``) predicted as any
    abnormal class; Sp = fraction of true-normal cycles predicted normal. Se therefore
    credits a true-crackle predicted as wheeze, since both are abnormal: the metric measures
    normal-vs-abnormal discrimination rather than 4-way accuracy. Report per-class Se too.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    is_ab_t = y_true != normal_label
    is_ab_p = y_pred != normal_label
    Se = is_ab_p[is_ab_t].sum() / is_ab_t.sum() if is_ab_t.sum() else 0.0
    Sp = (y_pred[~is_ab_t] == normal_label).sum() / (~is_ab_t).sum() if (~is_ab_t).sum() else 0.0
    return {"ICBHI_Score": (Se + Sp) / 2, "Se": float(Se), "Sp": float(Sp)}


def per_class_se(y_true, y_pred, labels):
    """Per-class recall (sensitivity) as a {label: recall} dict."""
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {lab: float(r) for lab, r in zip(labels, rec)}


def macro_f1(y_true, y_pred):
    """Macro-averaged F1 (unweighted mean over classes)."""
    return float(f1_score(y_true, y_pred, average="macro"))


def accuracy(y_true, y_pred):
    """Plain accuracy (secondary; never the headline metric on imbalanced data)."""
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def assert_not_degenerate(y_true, y_pred, labels):
    """Raise AssertionError when predictions occupy fewer than 2 confusion columns.

    A model that always predicts one class (e.g. always-normal on imbalanced lung) uses a
    single predicted-class column. Returns the number of columns used otherwise.
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
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.grid(False)
    _ = title
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cols_used
