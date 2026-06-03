"""Classical training and evaluation for the heart/lung sound"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time

from src import config

import numpy as np

import xgboost
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.metrics import (
    majority_vote,
    heart_macc,
    icbhi_score,
    per_class_se,
    macro_f1,
    accuracy,
    save_cm,
)
from src.split import assert_no_patient_leakage

__all__ = [
    "build_pipeline",
    "build_search",
    "tune_pipeline",
    "run_experiments",
    "MODEL_NAMES",
    "FEATURE_SETS",
    "SVM_GRID",
    "XGB_GRID",
]

SEED = getattr(config, "SEED", 42)
RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

MODEL_NAMES = ["logreg", "svm", "rf", "xgb"]

FEATURE_SETS = [
    ("A", "X_A", "A_mfcc_delta"),
    ("B", "X_B", "B_mfcc_delta_spectral"),
]

SVM_GRID = {"clf__C": [1, 10], "clf__gamma": ["scale", 0.01]}
XGB_GRID = {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 6]}

SVM_TUNE_MAX_WINDOWS = 7000

LUNG_NORMAL_LABEL = 3
LUNG_LABELS = [0, 1, 2, 3]
HEART_LABELS = [0, 1]


def build_pipeline(model_name, n_classes, y_train=None, seed=SEED):
    """Return a ``Pipeline([("scaler", StandardScaler()), ("clf","""
    name = model_name.lower()
    if name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    elif name == "svm":
        clf = SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=seed
        )
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    elif name == "xgb":
        if n_classes <= 2:
            spw = 1.0
            if y_train is not None:
                y_arr = np.asarray(y_train)
                n_pos = int((y_arr == 1).sum())
                n_neg = int((y_arr == 0).sum())
                spw = n_neg / max(n_pos, 1)
            clf = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                scale_pos_weight=spw,
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )
        else:
            clf = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {MODEL_NAMES}.")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _grouped_cv(n_classes, n_splits=3):
    """Return a patient-grouped CV: StratifiedGroupKFold, with"""
    if n_classes <= 2:
        return GroupKFold(n_splits=n_splits)
    return StratifiedGroupKFold(n_splits=n_splits)


def build_search(model_name, n_classes, y_train=None, seed=SEED):
    """Wrap the svm/xgb pipeline in a ``GridSearchCV`` over a"""
    name = model_name.lower()
    if name == "svm":
        grid = SVM_GRID
    elif name == "xgb":
        grid = XGB_GRID
    else:
        raise ValueError(
            f"build_search only tunes svm/xgb (logreg/rf use fixed defaults); got '{model_name}'."
        )
    pipe = build_pipeline(name, n_classes, y_train=y_train, seed=seed)
    cv = _grouped_cv(n_classes, n_splits=3)
    return GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )


tune_pipeline = build_search


def _subsample_groups(X, y, groups, cap, seed=SEED):
    """Patient-grouped, seeded train-only sub-sample capping the"""
    n = X.shape[0]
    if n <= cap:
        return X, y, groups
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(map(str, groups))), dtype=object)
    rng.shuffle(uniq)
    g_str = np.asarray(list(map(str, groups)), dtype=object)
    keep_mask = np.zeros(n, dtype=bool)
    kept = 0
    for g in uniq:
        gmask = g_str == g
        keep_mask |= gmask
        kept += int(gmask.sum())
        if kept >= cap:
            break
    if len(set(np.asarray(y)[keep_mask])) < 2:
        return X, y, groups
    return X[keep_mask], np.asarray(y)[keep_mask], np.asarray(groups)[keep_mask]


def _fit_tuned(model_name, n_classes, X_train, y_train, train_groups, seed=SEED):
    """Tune svm/xgb via grouped GridSearchCV (groups passed to"""
    name = model_name.lower()

    if name == "svm":
        Xs, ys, gs = _subsample_groups(
            X_train, y_train, train_groups, SVM_TUNE_MAX_WINDOWS, seed=seed
        )
        search = build_search("svm", n_classes, y_train=ys, seed=seed)
        try:
            search.fit(Xs, ys, groups=gs)
        except ValueError:
            search.cv = GroupKFold(n_splits=3)
            search.fit(Xs, ys, groups=gs)
        best_params = search.best_params_
        final = build_pipeline("svm", n_classes, y_train=y_train, seed=seed)
        final.set_params(**best_params)
        final.fit(X_train, y_train)
        return final, best_params

    search = build_search("xgb", n_classes, y_train=y_train, seed=seed)
    fit_kw = {"groups": train_groups}
    if n_classes > 2:
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        fit_kw["clf__sample_weight"] = sw
    try:
        search.fit(X_train, y_train, **fit_kw)
    except ValueError:
        search.cv = GroupKFold(n_splits=3)
        search.fit(X_train, y_train, **fit_kw)
    return search.best_estimator_, search.best_params_


def _fit_fixed(model_name, n_classes, X_train, y_train, seed=SEED):
    """Fit a model with fixed defaults (no grid), applying xgb"""
    pipe = build_pipeline(model_name, n_classes, y_train=y_train, seed=seed)
    if model_name.lower() == "xgb" and n_classes > 2:
        sw = compute_sample_weight(class_weight="balanced", y=y_train)
        pipe.fit(X_train, y_train, clf__sample_weight=sw)
    else:
        pipe.fit(X_train, y_train)
    return pipe


def _abnormal_proba(estimator, X, classes):
    """Per-row abnormal-class probability for the heart AUC,"""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        classes = np.asarray(classes)
        if 1 in classes:
            return proba[:, int(np.where(classes == 1)[0][0])]
        return proba[:, -1]
    if hasattr(estimator, "decision_function"):
        df = np.asarray(estimator.decision_function(X), dtype=float)
        lo, hi = df.min(), df.max()
        return (df - lo) / (hi - lo) if hi > lo else np.zeros_like(df)
    return np.asarray(estimator.predict(X), dtype=float)


def run_experiments(modality, cache_dict, figures_dir=FIGURES_DIR):
    """Run the 8 (feature_set x model) experiments for"""
    os.makedirs(figures_dir, exist_ok=True)

    n_classes = 2 if modality == "heart" else 4
    split_arr = np.asarray(cache_dict["split"], dtype=object)
    labels = np.asarray(cache_dict["labels"], dtype=int)
    pid = np.asarray(list(map(str, cache_dict["patient_id"])), dtype=object)
    rec = np.asarray(list(map(str, cache_dict["recording_id"])), dtype=object)

    tr = split_arr == "train"
    te = split_arr == "test"

    assert_no_patient_leakage(pid[tr], pid[te])

    y_train = labels[tr]
    y_test = labels[te]
    groups_train = pid[tr]
    rec_test = rec[te]

    n_train_segments = int(tr.sum())
    n_test_segments = int(te.sum())
    n_train_recordings = len(set(rec[tr]))
    n_test_recordings = len(set(rec[te]))
    n_train_patients = len(set(pid[tr]))
    n_test_patients = len(set(pid[te]))

    rows = []
    for fs_id, cache_key, fs_label in FEATURE_SETS:
        X = np.asarray(cache_dict[cache_key], dtype="float64")
        X_train, X_test = X[tr], X[te]

        for model_name in MODEL_NAMES:
            t0 = time.perf_counter()
            if model_name in ("svm", "xgb"):
                estimator, best_params = _fit_tuned(
                    model_name, n_classes, X_train, y_train, groups_train, seed=SEED
                )
            else:
                estimator = _fit_fixed(model_name, n_classes, X_train, y_train, seed=SEED)
                best_params = {}
            train_time_s = time.perf_counter() - t0

            preds = np.asarray(estimator.predict(X_test))
            classes = getattr(estimator, "classes_", np.unique(y_train))

            if modality == "heart":
                pred_rec = majority_vote(preds, rec_test)
                true_rec = (
                    majority_vote(y_test, rec_test).reindex(pred_rec.index)
                )
                win_score = _abnormal_proba(estimator, X_test, classes)
                import pandas as pd

                score_rec = (
                    pd.Series(win_score, index=rec_test)
                    .groupby(level=0)
                    .mean()
                    .reindex(pred_rec.index)
                )
                y_true_rec = true_rec.to_numpy().astype(int)
                y_pred_rec = pred_rec.to_numpy().astype(int)
                m = heart_macc(y_true_rec, y_pred_rec, y_score_rec=score_rec.to_numpy())
                cm_png = os.path.join(
                    figures_dir, f"cm_heart_{fs_id}_{model_name}.png"
                )
                save_cm(
                    y_true_rec, y_pred_rec, HEART_LABELS,
                    f"heart {fs_label} {model_name} (recording-level)", cm_png,
                )
                row = {
                    "modality": "heart",
                    "feature_set": fs_label,
                    "model": model_name,
                    "primary_metric_name": "MAcc",
                    "primary_metric": float(m["MAcc"]),
                    "MAcc": float(m["MAcc"]),
                    "Se": float(m["Se"]),
                    "Sp": float(m["Sp"]),
                    "macro_f1": float(m["macro_f1"]),
                    "auc_roc": float(m.get("auc_roc", float("nan"))),
                    "accuracy": float(m["accuracy"]),
                    "n_train": n_train_segments,
                    "n_test": n_test_recordings,
                    "best_params": str(best_params),
                    "cm_figure": os.path.basename(cm_png),
                }
            else:
                m = icbhi_score(y_test, preds, normal_label=LUNG_NORMAL_LABEL)
                pcs = per_class_se(y_test, preds, LUNG_LABELS)
                cm_png = os.path.join(
                    figures_dir, f"cm_lung_{fs_id}_{model_name}.png"
                )
                save_cm(
                    y_test, preds, LUNG_LABELS,
                    f"lung {fs_label} {model_name} (cycle-level)", cm_png,
                )
                row = {
                    "modality": "lung",
                    "feature_set": fs_label,
                    "model": model_name,
                    "primary_metric_name": "ICBHI_Score",
                    "primary_metric": float(m["ICBHI_Score"]),
                    "ICBHI_Score": float(m["ICBHI_Score"]),
                    "Se": float(m["Se"]),
                    "Sp": float(m["Sp"]),
                    "macro_f1": float(macro_f1(y_test, preds)),
                    "auc_roc": "",
                    "accuracy": float(accuracy(y_test, preds)),
                    "se_crackle": pcs[0],
                    "se_wheeze": pcs[1],
                    "se_both": pcs[2],
                    "se_normal": pcs[3],
                    "n_train": n_train_segments,
                    "n_test": n_test_segments,
                    "best_params": str(best_params),
                    "cm_figure": os.path.basename(cm_png),
                }

            row.update(
                {
                    "train_time_s": float(train_time_s),
                    "n_train_segments": n_train_segments,
                    "n_test_segments": n_test_segments,
                    "n_train_recordings": n_train_recordings,
                    "n_test_recordings": n_test_recordings,
                    "n_train_patients": n_train_patients,
                    "n_test_patients": n_test_patients,
                }
            )
            rows.append(row)
            print(
                f"  [{modality} {fs_label} {model_name}] "
                f"{row['primary_metric_name']}={row['primary_metric']:.4f} "
                f"Se={row['Se']:.3f} Sp={row['Sp']:.3f} "
                f"t={train_time_s:.1f}s params={best_params}"
            )

    return rows
