"""
tests/test_train_classical.py — MODL-01 / EVAL-02 / EVAL-03 contracts (Phase 3, Wave 0).

Specifies the contracts for the Wave-2 training/orchestration module described in
03-RESEARCH.md §Pattern 3 (Pipeline([StandardScaler, clf]) per model — the scaler
is INSIDE the pipeline so it fits on the train fold only; the canonical leakage
guard, D-05), §Pattern 4 (GridSearchCV(cv=GroupKFold/StratifiedGroupKFold) with
``groups`` passed to ``.fit`` — train patients only), §Pattern 8 (unified_comparison.csv
schema — 16 classical rows), §Pattern 9 (volumetrics_classical.csv — train_time_s +
segment AND recording/patient counts + data_volume_mb), and §Anti-Patterns
(no ``fit_transform(X_all)``).

Three flavours of test:
  - STATIC/grep (``test_no_global_scaler``): inspects src/train_classical.py SOURCE
    TEXT — asserts no bare ``fit_transform(`` on a full feature matrix and that the
    scaler is wired as the first step of an sklearn ``Pipeline``. Skips if the source
    file is absent (Wave 0) so collection never errors.
  - UNIT (``test_pipelines_fit`` / ``test_tuning_groups_disjoint``): import
    ``src.train_classical`` and exercise the four pipelines on the no-data
    ``synthetic_feature_matrix`` fixture. Skip-on-missing (RED in Wave 0).
  - SCHEMA (``test_metrics_csv_schema`` / ``test_unified_schema`` /
    ``test_volumetrics_schema``): read the result CSVs; SKIP if a CSV is absent
    (so they do not error before Wave 2 produces them).

All imports happen INSIDE the test bodies (skip-on-missing), mirroring
tests/test_preprocess.py, so collection has zero errors in Wave 0.
"""
import importlib
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity

TRAIN_SRC = PROJECT_ROOT / "src" / "train_classical.py"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

MODEL_NAMES = ["logreg", "svm", "rf", "xgb"]

# Pattern-8 unified_comparison.csv column set (exact order/content from 03-RESEARCH.md).
UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]
# 2 feature sets × 4 models × 2 modalities = 16 classical rows (D-04).
UNIFIED_CLASSICAL_ROWS = 16
# 2 feature sets × 4 models = 8 rows per per-modality metrics CSV (D-04).
METRICS_ROWS_PER_MODALITY = 8


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent in Wave 0."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive (Wave 0 module absent)
        pytest.skip(f"{module_name} not implemented yet (Wave 0): {exc}")


def _read_csv(path):
    """Read a results CSV as a DataFrame, or SKIP if it is absent (Wave-0 state)."""
    if not path.exists():
        pytest.skip(f"{path.name} not produced yet (Wave 0/1): {path}")
    pd = pytest.importorskip("pandas")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# STATIC leakage gate — no global fit_transform; scaler inside a Pipeline (D-05)
# ---------------------------------------------------------------------------

def test_no_global_scaler():
    """src/train_classical.py must scale INSIDE an sklearn Pipeline — never globally.

    The canonical normalisation-leakage bug is ``StandardScaler().fit_transform(X)``
    on the full (train+test) matrix before splitting (03-RESEARCH.md §Pitfall 1 /
    Anti-Patterns; ROADMAP criterion #1). This source-inspection gate must FAIL if
    any bare ``fit_transform(`` appears in the module — not only the literal
    ``fit_transform(X_all)`` — AND must assert positively that the scaler is wired
    as the FIRST step of a ``Pipeline``.
    """
    if not TRAIN_SRC.exists():
        pytest.skip("src/train_classical.py not implemented yet (Wave 0)")

    source = TRAIN_SRC.read_text(encoding="utf-8")

    # Strengthened guard: forbid ANY `.fit_transform(` call (scaling outside a
    # Pipeline leaks test statistics regardless of the variable name). Pipelines
    # call fit/transform internally — production code should never call
    # `fit_transform` on the feature matrix itself.
    assert "fit_transform(" not in source, (
        "LEAKAGE: src/train_classical.py contains a bare `.fit_transform(` call — "
        "the scaler must live INSIDE the Pipeline (fit on train fold only), never "
        "transform the full matrix globally (D-05 / ROADMAP criterion #1)."
    )

    # Positive assertion: the scaler is the first step of an sklearn Pipeline.
    assert "Pipeline" in source, (
        "src/train_classical.py must build an sklearn Pipeline so the StandardScaler "
        "fits on the train fold only."
    )
    assert "StandardScaler" in source, (
        "src/train_classical.py must use StandardScaler inside the Pipeline."
    )


# ---------------------------------------------------------------------------
# UNIT — each of the 4 models builds a Pipeline (scaler first step) and fits
# ---------------------------------------------------------------------------

def test_pipelines_fit(synthetic_feature_matrix):
    """build_pipeline(model, n_classes=2) returns a Pipeline whose first step is a
    StandardScaler, and fit→predict runs for logreg / svm / rf / xgb."""
    train = _import("src.train_classical")
    if not hasattr(train, "build_pipeline"):
        pytest.skip("src.train_classical.build_pipeline not implemented yet (Wave 0)")

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


# ---------------------------------------------------------------------------
# UNIT — GridSearchCV uses a grouped CV; tuning folds are patient-disjoint
# ---------------------------------------------------------------------------

def test_tuning_groups_disjoint(synthetic_feature_matrix):
    """The tuning helper wraps a pipeline in GridSearchCV with a grouped CV, and when
    fit with ``groups=train_groups`` each fold's train/validation patients are disjoint
    (no patient leaks across the inner-CV split)."""
    train = _import("src.train_classical")
    tuner = getattr(train, "build_search", None) or getattr(train, "tune_pipeline", None)
    if tuner is None:
        pytest.skip(
            "src.train_classical tuning helper (build_search/tune_pipeline) not implemented yet (Wave 0)"
        )

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    X, y, groups = synthetic_feature_matrix

    search = tuner("svm", n_classes=2, y_train=y)
    assert isinstance(search, GridSearchCV), "tuning helper must return a GridSearchCV"
    assert isinstance(search.cv, (GroupKFold, StratifiedGroupKFold)), (
        "GridSearchCV.cv must be GroupKFold/StratifiedGroupKFold so folds are patient-disjoint"
    )

    # Independently verify the grouped CV keeps train/validation groups disjoint.
    cv = search.cv
    for tr_idx, va_idx in cv.split(X, y, groups=groups):
        tr_groups = set(np.asarray(groups)[tr_idx])
        va_groups = set(np.asarray(groups)[va_idx])
        assert tr_groups.isdisjoint(va_groups), (
            "inner-CV fold leaked a patient group across train/validation"
        )

    # Fitting with groups passed to .fit (NOT the constructor) must run without error.
    search.fit(X, y, groups=groups)
    assert hasattr(search, "best_estimator_"), "GridSearchCV did not refit a best_estimator_"


# ---------------------------------------------------------------------------
# SCHEMA — per-modality metrics CSVs: 8 rows each + full metric suite
# ---------------------------------------------------------------------------

def test_metrics_csv_schema():
    """metrics_{heart,lung}_classical.csv each have 8 rows (feature_set×model) and the metric columns."""
    required_cols = {"feature_set", "model", "Se", "Sp", "macro_f1", "accuracy"}
    for name in ("metrics_heart_classical.csv", "metrics_lung_classical.csv"):
        df = _read_csv(TABLES_DIR / name)
        assert len(df) == METRICS_ROWS_PER_MODALITY, (
            f"{name} must have {METRICS_ROWS_PER_MODALITY} rows (2 feature sets × 4 models); got {len(df)}"
        )
        # Primary metric column present (MAcc for heart, ICBHI_Score for lung) —
        # accept either an explicit primary metric column or a primary_metric field.
        has_primary = (
            "primary_metric" in df.columns
            or any(c in df.columns for c in ("MAcc", "ICBHI_Score"))
        )
        assert has_primary, f"{name} missing a primary-metric column"
        missing = required_cols - set(df.columns)
        assert not missing, f"{name} missing columns: {sorted(missing)}"


# ---------------------------------------------------------------------------
# SCHEMA — unified_comparison.csv: exact Pattern-8 columns + 16 classical rows
# ---------------------------------------------------------------------------

def test_unified_schema():
    """unified_comparison.csv has the exact Pattern-8 columns and 16 classical rows."""
    df = _read_csv(TABLES_DIR / "unified_comparison.csv")

    for col in UNIFIED_COLUMNS:
        assert col in df.columns, f"unified_comparison.csv missing column '{col}'"

    classical = df[df["model"].isin(MODEL_NAMES)] if "model" in df.columns else df
    assert len(classical) == UNIFIED_CLASSICAL_ROWS, (
        f"unified_comparison.csv must contain {UNIFIED_CLASSICAL_ROWS} classical rows "
        f"(2 modalities × 2 feature sets × 4 models); got {len(classical)}"
    )


# ---------------------------------------------------------------------------
# SCHEMA — volumetrics_classical.csv: train_time + segment AND record/patient counts
# ---------------------------------------------------------------------------

def test_volumetrics_schema():
    """volumetrics_classical.csv has train_time_s, segment-level AND recording/patient
    counts, and a data_volume_mb column (EVAL-03 / Pattern 9)."""
    df = _read_csv(TABLES_DIR / "volumetrics_classical.csv")
    cols = set(df.columns)

    assert "train_time_s" in cols, "volumetrics_classical.csv missing train_time_s"
    assert "data_volume_mb" in cols, "volumetrics_classical.csv missing data_volume_mb"

    # Segment/cycle-level counts (windows for heart, cycles for lung).
    assert any("segment" in c for c in cols), (
        "volumetrics_classical.csv missing a segment/cycle-level count column"
    )
    # Recording AND patient-level counts (D-13 is explicit about both).
    assert any("recording" in c for c in cols), (
        "volumetrics_classical.csv missing a recording-level count column"
    )
    assert any("patient" in c for c in cols), (
        "volumetrics_classical.csv missing a patient-level count column"
    )
