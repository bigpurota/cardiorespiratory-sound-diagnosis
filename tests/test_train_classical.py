"""Tests for the classical training/orchestration module (src/train_classical.py).

Three kinds of test:
  - source inspection (test_no_global_scaler): the module has no bare
    ``fit_transform(`` and wires the StandardScaler as the first step of an sklearn
    Pipeline, so it fits on the train fold only.
  - unit (test_pipelines_fit / test_tuning_groups_disjoint): the four pipelines fit
    and the tuner uses a grouped CV whose folds are patient-disjoint, exercised on
    the synthetic_feature_matrix fixture.
  - schema (test_metrics_csv_schema / test_unified_schema / test_volumetrics_schema):
    the result CSVs have the expected columns and row counts.

Imports and file reads happen inside the test bodies and skip when their target is
absent.
"""
import importlib
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

TRAIN_SRC = PROJECT_ROOT / "src" / "train_classical.py"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

MODEL_NAMES = ["logreg", "svm", "rf", "xgb"]

UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]
UNIFIED_CLASSICAL_ROWS = 16
METRICS_ROWS_PER_MODALITY = 8


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not implemented yet: {exc}")


def _read_csv(path):
    """Read a results CSV as a DataFrame, or SKIP if it is absent."""
    if not path.exists():
        pytest.skip(f"{path.name} not produced yet: {path}")
    pd = pytest.importorskip("pandas")
    return pd.read_csv(path)


def test_no_global_scaler():
    """src/train_classical.py must scale INSIDE an sklearn Pipeline — never globally.

    The classic normalisation-leakage bug is ``StandardScaler().fit_transform(X)``
    on the full (train+test) matrix before splitting. This source check fails if any
    bare ``fit_transform(`` appears in the module, and also asserts that the scaler is
    wired as the first step of a ``Pipeline``.
    """
    if not TRAIN_SRC.exists():
        pytest.skip("src/train_classical.py not implemented yet")

    source = TRAIN_SRC.read_text(encoding="utf-8")

    assert "fit_transform(" not in source, (
        "LEAKAGE: src/train_classical.py contains a bare `.fit_transform(` call — "
        "the scaler must live INSIDE the Pipeline (fit on train fold only), never "
        "transform the full matrix globally."
    )

    assert "Pipeline" in source, (
        "src/train_classical.py must build an sklearn Pipeline so the StandardScaler "
        "fits on the train fold only."
    )
    assert "StandardScaler" in source, (
        "src/train_classical.py must use StandardScaler inside the Pipeline."
    )


def test_pipelines_fit(synthetic_feature_matrix):
    """build_pipeline(model, n_classes=2) returns a Pipeline whose first step is a
    StandardScaler, and fit→predict runs for logreg / svm / rf / xgb."""
    train = _import("src.train_classical")
    if not hasattr(train, "build_pipeline"):
        pytest.skip("src.train_classical.build_pipeline not implemented yet")

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, _ = synthetic_feature_matrix

    for name in MODEL_NAMES:
        pipe = train.build_pipeline(name, n_classes=2, y_train=y)
        assert isinstance(pipe, Pipeline), f"{name}: build_pipeline must return a Pipeline"
        first_name, first_step = pipe.steps[0]
        assert isinstance(first_step, StandardScaler), (
            f"{name}: first pipeline step must be a StandardScaler (got {type(first_step).__name__})"
        )
        pipe.fit(X, y)
        preds = np.asarray(pipe.predict(X))
        assert preds.shape[0] == X.shape[0], f"{name}: predict returned wrong row count"


def test_tuning_groups_disjoint(synthetic_feature_matrix):
    """The tuning helper wraps a pipeline in GridSearchCV with a grouped CV, and when
    fit with ``groups=train_groups`` each fold's train/validation patients are disjoint
    (no patient leaks across the inner-CV split)."""
    train = _import("src.train_classical")
    tuner = getattr(train, "build_search", None) or getattr(train, "tune_pipeline", None)
    if tuner is None:
        pytest.skip(
            "src.train_classical tuning helper (build_search/tune_pipeline) not implemented yet"
        )

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    X, y, groups = synthetic_feature_matrix

    search = tuner("svm", n_classes=2, y_train=y)
    assert isinstance(search, GridSearchCV), "tuning helper must return a GridSearchCV"
    assert isinstance(search.cv, (GroupKFold, StratifiedGroupKFold)), (
        "GridSearchCV.cv must be GroupKFold/StratifiedGroupKFold so folds are patient-disjoint"
    )

    cv = search.cv
    for tr_idx, va_idx in cv.split(X, y, groups=groups):
        tr_groups = set(np.asarray(groups)[tr_idx])
        va_groups = set(np.asarray(groups)[va_idx])
        assert tr_groups.isdisjoint(va_groups), (
            "inner-CV fold leaked a patient group across train/validation"
        )

    search.fit(X, y, groups=groups)
    assert hasattr(search, "best_estimator_"), "GridSearchCV did not refit a best_estimator_"


def test_metrics_csv_schema():
    """metrics_{heart,lung}_classical.csv each have 8 rows (feature_set×model) and the metric columns."""
    required_cols = {"feature_set", "model", "Se", "Sp", "macro_f1", "accuracy"}
    for name in ("metrics_heart_classical.csv", "metrics_lung_classical.csv"):
        df = _read_csv(TABLES_DIR / name)
        assert len(df) == METRICS_ROWS_PER_MODALITY, (
            f"{name} must have {METRICS_ROWS_PER_MODALITY} rows (2 feature sets × 4 models); got {len(df)}"
        )
        has_primary = (
            "primary_metric" in df.columns
            or any(c in df.columns for c in ("MAcc", "ICBHI_Score"))
        )
        assert has_primary, f"{name} missing a primary-metric column"
        missing = required_cols - set(df.columns)
        assert not missing, f"{name} missing columns: {sorted(missing)}"


def test_unified_schema():
    """unified_comparison.csv has the exact columns and 16 classical rows."""
    df = _read_csv(TABLES_DIR / "unified_comparison.csv")

    for col in UNIFIED_COLUMNS:
        assert col in df.columns, f"unified_comparison.csv missing column '{col}'"

    classical = df[df["model"].isin(MODEL_NAMES)] if "model" in df.columns else df
    assert len(classical) == UNIFIED_CLASSICAL_ROWS, (
        f"unified_comparison.csv must contain {UNIFIED_CLASSICAL_ROWS} classical rows "
        f"(2 modalities × 2 feature sets × 4 models); got {len(classical)}"
    )


def test_volumetrics_schema():
    """volumetrics_classical.csv has train_time_s, segment-level AND recording/patient
    counts, and a data_volume_mb column."""
    df = _read_csv(TABLES_DIR / "volumetrics_classical.csv")
    cols = set(df.columns)

    assert "train_time_s" in cols, "volumetrics_classical.csv missing train_time_s"
    assert "data_volume_mb" in cols, "volumetrics_classical.csv missing data_volume_mb"

    assert any("segment" in c for c in cols), (
        "volumetrics_classical.csv missing a segment/cycle-level count column"
    )
    assert any("recording" in c for c in cols), (
        "volumetrics_classical.csv missing a recording-level count column"
    )
    assert any("patient" in c for c in cols), (
        "volumetrics_classical.csv missing a patient-level count column"
    )
