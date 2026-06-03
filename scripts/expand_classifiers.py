"""Expand the classical classifier panel from 4 to 9 and re-test rank transfer.

The cross-modal "rankings do not transfer" claim originally rested on four
classifiers (n = 4 for the Spearman rank correlation). This script adds five more
class-weighted classifiers (Extra-Trees, AdaBoost, Ridge, SGD-logistic, LinearSVC)
on both modalities and both feature sets, recomputes the per-modality rankings, and
re-estimates the heart-vs-lung rank correlation on the larger panel (n = 9 per
feature set). The original four scores are read from the committed metrics CSVs so
the slow RBF-SVM is not refitted; the five new ones are fit here on the local
feature caches. Deterministic (seed 42), CPU only, all single-threaded.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.metrics import majority_vote, heart_macc, icbhi_score

ROOT = os.path.dirname(config.RESULTS_DIR)
HEART = os.path.join(ROOT, "features", "heart_classical.npy")
LUNG = os.path.join(ROOT, "features", "lung_classical.npy")
TAB = os.path.join(config.RESULTS_DIR, "tables")
SEED = 42

FEATURE_SETS = [("A", "X_A", "A_mfcc_delta"), ("B", "X_B", "B_mfcc_delta_spectral")]


def new_models():
    """Five additional classifiers, all class-weight aware so the comparison
    stays consistent with the report's in-fold reweighting discipline."""
    return {
        "extratrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=SEED, n_jobs=1),
        "adaboost":   AdaBoostClassifier(n_estimators=300, random_state=SEED),  # sample_weight at fit
        "ridge":      RidgeClassifier(class_weight="balanced", random_state=SEED),
        "sgd_log":    SGDClassifier(loss="log_loss", class_weight="balanced",
                                    max_iter=2000, random_state=SEED),
        "linsvc":     LinearSVC(class_weight="balanced", dual="auto",
                                max_iter=5000, random_state=SEED),
    }


def load(path):
    return np.load(path, allow_pickle=True).item()


def fit_score(modality, cache, model_name, model, cache_key):
    split = np.asarray(cache["split"], dtype=object)
    labels = np.asarray(cache["labels"], dtype=int)
    rec = np.asarray(list(map(str, cache["recording_id"])), dtype=object)
    tr, te = split == "train", split == "test"
    X = np.asarray(cache[cache_key], dtype="float64")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    if model_name == "adaboost":
        sw = compute_sample_weight("balanced", labels[tr])
        pipe.fit(X[tr], labels[tr], clf__sample_weight=sw)
    else:
        pipe.fit(X[tr], labels[tr])
    pred = pipe.predict(X[te])
    if modality == "heart":
        pr = majority_vote(pred, rec[te])
        tr_rec = majority_vote(labels[te], rec[te]).reindex(pr.index)
        return float(heart_macc(tr_rec.to_numpy().astype(int), pr.to_numpy().astype(int))["MAcc"])
    return float(icbhi_score(labels[te], pred, normal_label=3)["ICBHI_Score"])


def existing_scores():
    """Pull the original four classifiers' primary metric from committed CSVs."""
    h = pd.read_csv(os.path.join(TAB, "metrics_heart_classical.csv"))
    l = pd.read_csv(os.path.join(TAB, "metrics_lung_classical.csv"))
    out = {}
    for fs_label in ("A_mfcc_delta", "B_mfcc_delta_spectral"):
        for m in ("logreg", "svm", "rf", "xgb"):
            hv = h[(h.feature_set == fs_label) & (h.model == m)].primary_metric.iloc[0]
            lv = l[(l.feature_set == fs_label) & (l.model == m)].primary_metric.iloc[0]
            out[(fs_label, m)] = (float(hv), float(lv))
    return out


def run():
    heart, lung = load(HEART), load(LUNG)
    base = existing_scores()
    rows = []
    for fs_id, key, fs_label in FEATURE_SETS:
        # original 4
        for m in ("logreg", "svm", "rf", "xgb"):
            hv, lv = base[(fs_label, m)]
            rows.append(dict(feature_set=fs_id, model=m, heart=hv, lung=lv, source="original"))
        # new 5
        for name, mdl in new_models().items():
            hv = fit_score("heart", heart, name, mdl, key)
            mdl2 = new_models()[name]  # fresh instance for lung
            lv = fit_score("lung", lung, name, mdl2, key)
            rows.append(dict(feature_set=fs_id, model=name, heart=round(hv, 4),
                             lung=round(lv, 4), source="new"))
            print(f"[{fs_id} {name}] heart MAcc={hv:.4f}  lung ICBHI={lv:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "expanded_classifiers.csv"), index=False)

    print("\n=== Spearman rank correlation heart-MAcc vs lung-ICBHI (n=9 per set) ===")
    sp_rows = []
    for fs_id in ("A", "B"):
        sub = df[df.feature_set == fs_id]
        rho, p = spearmanr(sub.heart, sub.lung)
        print(f"  set {fs_id}: rho={rho:+.3f}  p={p:.3f}  (n={len(sub)})")
        sp_rows.append(dict(feature_set=fs_id, n=len(sub), rho=round(rho, 3), p=round(p, 4)))
    rho, p = spearmanr(df.heart, df.lung)
    print(f"  pooled: rho={rho:+.3f}  p={p:.3f}  (n={len(df)})")
    sp_rows.append(dict(feature_set="pooled", n=len(df), rho=round(rho, 3), p=round(p, 4)))
    pd.DataFrame(sp_rows).to_csv(os.path.join(TAB, "expanded_spearman.csv"), index=False)

    print("\n=== per-modality rankings (set B, best to worst) ===")
    b = df[df.feature_set == "B"]
    print("HEART:", "  >  ".join(b.sort_values("heart", ascending=False).model))
    print("LUNG :", "  >  ".join(b.sort_values("lung", ascending=False).model))
    print(f"\nwrote {TAB}/expanded_classifiers.csv and expanded_spearman.csv")


if __name__ == "__main__":
    run()
