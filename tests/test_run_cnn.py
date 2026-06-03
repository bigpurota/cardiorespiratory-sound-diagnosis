"""Schema tests for the deep-learning merge into"""
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
UNIFIED_CSV = TABLES_DIR / "unified_comparison.csv"

UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]

CLASSICAL_MODELS = ["logreg", "svm", "rf", "xgb"]
DL_MODELS = ["cnn", "effnet_b0"]
UNIFIED_CLASSICAL_ROWS = 16
UNIFIED_DL_ROWS = 4
UNIFIED_TOTAL_ROWS = 20


def _read_csv(path):
    """Read a results CSV as a DataFrame, or SKIP if it is absent."""
    if not path.exists():
        pytest.skip(f"{path.name} not produced yet: {path}")
    pd = pytest.importorskip("pandas")
    return pd.read_csv(path)


def _skip_if_not_dl_updated(df):
    """SKIP while the CSV still holds only the 16 classical rows."""
    if "model" not in df.columns:
        pytest.skip("unified_comparison.csv has no 'model' column yet")
    if not df["model"].isin(DL_MODELS).any():
        pytest.skip(
            "unified_comparison.csv not DL-updated yet (no cnn/effnet_b0 rows)"
        )


def test_unified_schema():
    """unified_comparison.csv columns == UNIFIED_COLUMNS in EXACT"""
    df = _read_csv(UNIFIED_CSV)
    _skip_if_not_dl_updated(df)

    assert list(df.columns) == UNIFIED_COLUMNS, (
        "unified_comparison.csv columns must equal UNIFIED_COLUMNS in exact order; "
        f"got {list(df.columns)}"
    )


def test_matrix_complete():
    """After the DL merge: cnn+effnet_b0 rows exist for BOTH"""
    df = _read_csv(UNIFIED_CSV)
    _skip_if_not_dl_updated(df)

    assert "modality" in df.columns, "unified_comparison.csv missing 'modality' column"

    for model in DL_MODELS:
        for modality in ("heart", "lung"):
            present = (
                (df["model"] == model) & (df["modality"] == modality)
            ).any()
            assert present, (
                f"missing DL experiment: model={model!r}, modality={modality!r}"
            )

    dl = df[df["model"].isin(DL_MODELS)]
    assert len(dl) == UNIFIED_DL_ROWS, (
        f"expected {UNIFIED_DL_ROWS} DL rows (2 models × 2 modalities); got {len(dl)}"
    )

    classical = df[df["model"].isin(CLASSICAL_MODELS)]
    assert len(classical) == UNIFIED_CLASSICAL_ROWS, (
        f"the {UNIFIED_CLASSICAL_ROWS} classical rows must survive the DL merge; "
        f"got {len(classical)}"
    )

    assert len(df) == UNIFIED_TOTAL_ROWS, (
        f"unified_comparison.csv must hold {UNIFIED_TOTAL_ROWS} long-format rows "
        f"(16 classical + 4 DL); got {len(df)}"
    )
