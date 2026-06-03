"""Paired McNemar tests between the top classical models on each modality.

Reuses the exact src.train_classical fitting paths so the per-item predictions
match the numbers reported in Chapter 3, then runs McNemar's paired test on
recording-level (heart) and cycle-level (lung) correctness. The point is the
ranking inversion: XGBoost is rank-1 on heart but falls behind SVM on lung, and
this script asks whether that reordering is visible at the prediction level
rather than only in the rounded aggregate scores.

Deterministic (seed 42); CPU only; no GPU or checkpoints needed.
"""
import os

# Force single-threaded everywhere: nested parallelism (GridSearchCV loky workers
# spawning OMP-threaded XGBoost) segfaults libomp on macOS. n_jobs has no effect
# on the fitted result, only on speed, so the predictions are identical to Ch.3.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.model_selection import GridSearchCV, GroupKFold

from src import config
from src.train_classical import (
    build_pipeline, _grouped_cv, _subsample_groups,
    _fit_fixed, SVM_GRID, XGB_GRID, SVM_TUNE_MAX_WINDOWS,
)
from sklearn.utils.class_weight import compute_sample_weight
from src.metrics import majority_vote, heart_macc, icbhi_score

FEAT = os.path.join(config.RESULTS_DIR, "..", "features")
HEART = os.path.join(os.path.dirname(config.RESULTS_DIR), "features", "heart_classical.npy")
LUNG = os.path.join(os.path.dirname(config.RESULTS_DIR), "features", "lung_classical.npy")
OUT = os.path.join(config.RESULTS_DIR, "tables", "mcnemar_classical.csv")

SEED = 42


def _load(path):
    return np.load(path, allow_pickle=True).item()


def _fit_tuned_serial(model_name, n_classes, X_train, y_train, groups, seed=SEED):
    """Single-threaded re-implementation of train_classical._fit_tuned.

    Same grids, same grouped CV, same SVM subsample cap -> deterministically
    identical best_params and final model as Chapter 3, but with n_jobs=1 so the
    macOS loky/libomp nested-parallelism segfault cannot occur."""
    name = model_name.lower()
    if name == "svm":
        Xs, ys, gs = _subsample_groups(X_train, y_train, groups, SVM_TUNE_MAX_WINDOWS, seed=seed)
        pipe = build_pipeline("svm", n_classes, y_train=ys, seed=seed)
        cv = _grouped_cv(n_classes, n_splits=3)
        search = GridSearchCV(pipe, SVM_GRID, cv=cv, scoring="balanced_accuracy", n_jobs=1, refit=True)
        try:
            search.fit(Xs, ys, groups=gs)
        except ValueError:
            search.cv = GroupKFold(n_splits=3)
            search.fit(Xs, ys, groups=gs)
        final = build_pipeline("svm", n_classes, y_train=y_train, seed=seed)
        final.set_params(**search.best_params_)
        final.fit(X_train, y_train)
        return final
    # xgb
    pipe = build_pipeline("xgb", n_classes, y_train=y_train, seed=seed)
    pipe.set_params(clf__n_jobs=1)
    cv = _grouped_cv(n_classes, n_splits=3)
    search = GridSearchCV(pipe, XGB_GRID, cv=cv, scoring="balanced_accuracy", n_jobs=1, refit=True)
    fit_kw = {"groups": groups}
    if n_classes > 2:
        fit_kw["clf__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train)
    try:
        search.fit(X_train, y_train, **fit_kw)
    except ValueError:
        search.cv = GroupKFold(n_splits=3)
        search.fit(X_train, y_train, **fit_kw)
    return search.best_estimator_


def _fit_predict(modality, cache, model_name, cache_key):
    """Fit one classical model the exact same way Chapter 3 did and return its
    test predictions (cycle-level array, aligned with the test mask)."""
    n_classes = 2 if modality == "heart" else 4
    split = np.asarray(cache["split"], dtype=object)
    labels = np.asarray(cache["labels"], dtype=int)
    pid = np.asarray(list(map(str, cache["patient_id"])), dtype=object)
    tr, te = split == "train", split == "test"
    X = np.asarray(cache[cache_key], dtype="float64")
    if model_name in ("svm", "xgb"):
        est = _fit_tuned_serial(model_name, n_classes, X[tr], labels[tr], pid[tr], seed=SEED)
    else:
        est = _fit_fixed(model_name, n_classes, X[tr], labels[tr], seed=SEED)
    return est.predict(X[te]), labels[te], te


def _heart_correct(cache, preds, y_test, te):
    """Recording-level correctness vector (majority vote per recording)."""
    rec = np.asarray(list(map(str, cache["recording_id"])), dtype=object)[te]
    pred_rec = majority_vote(preds, rec)
    true_rec = majority_vote(y_test, rec).reindex(pred_rec.index)
    macc = heart_macc(true_rec.to_numpy().astype(int), pred_rec.to_numpy().astype(int))
    correct = (pred_rec.to_numpy().astype(int) == true_rec.to_numpy().astype(int))
    return correct, pred_rec.index, macc["MAcc"]


def mcnemar(correct_a, correct_b):
    """McNemar on two boolean correctness vectors aligned item-for-item.
    Returns (a_only_right, b_only_right) discordant counts plus an exact-binomial
    and a continuity-corrected chi-square p-value."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    a_only = int(np.sum(a & ~b))   # A right, B wrong
    b_only = int(np.sum(~a & b))   # A wrong, B right
    n = a_only + b_only
    # exact two-sided binomial on the discordant pairs
    p_exact = binomtest(min(a_only, b_only), n, 0.5, alternative="two-sided").pvalue if n else 1.0
    # continuity-corrected chi-square
    if n:
        stat = (abs(a_only - b_only) - 1) ** 2 / n
        p_chi = float(chi2.sf(stat, 1))
    else:
        stat, p_chi = 0.0, 1.0
    return a_only, b_only, float(stat), p_exact, p_chi


def run():
    rows = []
    print("=== HEART (recording-level, 626 recordings) ===")
    cache = _load(HEART)
    pred_xgb, yte, te = _fit_predict("heart", cache, "xgb", "X_B")
    cx, idx_x, macc_x = _heart_correct(cache, pred_xgb, yte, te)
    pred_svm, _, _ = _fit_predict("heart", cache, "svm", "X_B")
    cs, idx_s, macc_s = _heart_correct(cache, pred_svm, yte, te)
    # align both correctness vectors on a common recording order
    common = pd.Index(idx_x).intersection(pd.Index(idx_s))
    cx = pd.Series(cx, index=idx_x).reindex(common).to_numpy()
    cs = pd.Series(cs, index=idx_s).reindex(common).to_numpy()
    xgb_only, svm_only, stat, pe, pc = mcnemar(cx, cs)  # A=XGB, B=SVM
    print(f"XGBoost-B MAcc={macc_x:.4f}   SVM-B MAcc={macc_s:.4f}   (report: 0.9025 / 0.8694)")
    print(f"discordant: XGB-only-right={xgb_only}  SVM-only-right={svm_only}  "
          f"chi2_cc={stat:.3f}  p_exact={pe:.4f}  p_chi={pc:.4f}  n_items={len(common)}")
    rows.append(dict(modality="heart", level="recording", metric="MAcc", n_items=len(common),
                     xgb="xgb_B", svm="svm_B", xgb_score=round(macc_x, 4),
                     svm_score=round(macc_s, 4), xgb_only_right=xgb_only, svm_only_right=svm_only,
                     chi2_cc=round(stat, 3), p_exact=round(pe, 6), p_chi=round(pc, 6)))

    print("\n=== LUNG (cycle-level, 2636 cycles) ===")
    cache = _load(LUNG)
    pred_xgb, yte, te = _fit_predict("lung", cache, "xgb", "X_B")
    icbhi_x = icbhi_score(yte, pred_xgb, normal_label=3)["ICBHI_Score"]
    cx = (pred_xgb == yte)
    pred_svm, yte2, _ = _fit_predict("lung", cache, "svm", "X_B")
    icbhi_s = icbhi_score(yte2, pred_svm, normal_label=3)["ICBHI_Score"]
    cs = (pred_svm == yte2)
    xgb_only, svm_only, stat, pe, pc = mcnemar(cx, cs)  # A=XGB, B=SVM
    print(f"XGBoost-B ICBHI={icbhi_x:.4f}   SVM-B ICBHI={icbhi_s:.4f}   (report: 0.5002 / 0.5368)")
    print(f"discordant (raw cycle accuracy): XGB-only-right={xgb_only}  SVM-only-right={svm_only}  "
          f"chi2_cc={stat:.3f}  p_exact={pe:.4f}  p_chi={pc:.4f}  n_items={len(yte)}")
    print("  note: McNemar is on RAW cycle accuracy; the balanced ICBHI score reverses the order.")
    rows.append(dict(modality="lung", level="cycle", metric="raw_accuracy", n_items=int(len(yte)),
                     xgb="xgb_B", svm="svm_B", xgb_score=round(float(icbhi_x), 4),
                     svm_score=round(float(icbhi_s), 4), xgb_only_right=xgb_only, svm_only_right=svm_only,
                     chi2_cc=round(stat, 3), p_exact=round(pe, 6), p_chi=round(pc, 6)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
