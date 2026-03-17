"""
tests/test_run_cnn.py — MODL-02 / SC4 contracts (Phase 4, Wave 0).

Schema-level contracts for the Wave-3 ``scripts/run_cnn.py`` orchestration that appends
the 4 deep-learning rows to ``results/tables/unified_comparison.csv`` (04-RESEARCH.md
§Code Examples 7). Mirrors ``tests/test_train_classical.py::test_unified_schema``:

  - ``test_unified_schema`` (SC4): the (DL-appended) CSV columns match ``UNIFIED_COLUMNS``
    in exact order — the same 12-column Pattern-8 header used by scripts/run_classical.py.
  - ``test_matrix_complete`` (SC4): after the DL merge, conceptual matrix experiments
    #5/#8/#9/#10 are present — rows with ``model in {"cnn","effnet_b0"}`` exist for BOTH
    ``modality=="heart"`` and ``modality=="lung"`` — AND the 16 classical rows survive
    (model in {logreg,svm,rf,xgb}), giving 20 physical long-format rows (A1 framing: do
    NOT collapse the classical rows to satisfy "10 experiments").

SCHEMA flavour: ``_read_csv`` SKIPS if the CSV has not yet been DL-updated (so these stubs
do not error before Wave 3), and they also SKIP cleanly while only the 16 classical rows
are present. Wave-0 collection has zero errors.
"""
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity

TABLES_DIR = PROJECT_ROOT / "results" / "tables"
UNIFIED_CSV = TABLES_DIR / "unified_comparison.csv"

# Pattern-8 unified_comparison.csv column set — IDENTICAL to scripts/run_classical.py
# UNIFIED_COLUMNS (exact order, 12 columns "modality" … "n_test"). VERIFIED from the
# live CSV header and 04-RESEARCH §Code Examples 7.
UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]

CLASSICAL_MODELS = ["logreg", "svm", "rf", "xgb"]
DL_MODELS = ["cnn", "effnet_b0"]
UNIFIED_CLASSICAL_ROWS = 16   # 2 modalities × 2 feature sets × 4 models (D-04)
UNIFIED_DL_ROWS = 4           # #5 cnn-heart, #8 cnn-lung, #9 effnet-heart, #10 effnet-lung
UNIFIED_TOTAL_ROWS = 20       # 16 classical + 4 DL (A1: long format, NOT collapsed to 10)


def _read_csv(path):
    """Read a results CSV as a DataFrame, or SKIP if it is absent (Wave-0 state)."""
    if not path.exists():
        pytest.skip(f"{path.name} not produced yet (Wave 0): {path}")
    pd = pytest.importorskip("pandas")
    return pd.read_csv(path)


def _skip_if_not_dl_updated(df):
    """SKIP while the CSV still holds only the 16 classical rows (pre-Wave-3 state)."""
    if "model" not in df.columns:
        pytest.skip("unified_comparison.csv has no 'model' column yet (Wave 0)")
    if not df["model"].isin(DL_MODELS).any():
        pytest.skip(
            "unified_comparison.csv not DL-updated yet (no cnn/effnet_b0 rows) — Wave 0/3"
        )


# ---------------------------------------------------------------------------
# SCHEMA — DL-appended unified_comparison.csv matches UNIFIED_COLUMNS exactly
# ---------------------------------------------------------------------------

def test_unified_schema():
    """unified_comparison.csv columns == UNIFIED_COLUMNS in EXACT order after the DL merge.

    The DL append (scripts/run_cnn.py) must rewrite the CSV in the SAME 12-column
    Pattern-8 order as scripts/run_classical.py — never adding/reordering columns.
    """
    df = _read_csv(UNIFIED_CSV)
    _skip_if_not_dl_updated(df)

    assert list(df.columns) == UNIFIED_COLUMNS, (
        "unified_comparison.csv columns must equal UNIFIED_COLUMNS in exact order; "
        f"got {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# SCHEMA — matrix complete: DL #5/#8/#9/#10 present + 16 classical rows survive
# ---------------------------------------------------------------------------

def test_matrix_complete():
    """After the DL merge: cnn+effnet_b0 rows exist for BOTH modalities; classical rows survive.

    Conceptual matrix experiments #5/#8/#9/#10 = (cnn,heart), (cnn,lung), (effnet_b0,heart),
    (effnet_b0,lung). All four must be present, AND the 16 classical rows (model in
    {logreg,svm,rf,xgb}) must remain → 20 physical long-format rows total (A1: do NOT
    collapse to 10).
    """
    df = _read_csv(UNIFIED_CSV)
    _skip_if_not_dl_updated(df)

    assert "modality" in df.columns, "unified_comparison.csv missing 'modality' column"

    # Each DL model must appear for BOTH heart and lung (the #5/#8/#9/#10 coverage).
    for model in DL_MODELS:
        for modality in ("heart", "lung"):
            present = (
                (df["model"] == model) & (df["modality"] == modality)
            ).any()
            assert present, (
                f"missing DL experiment: model={model!r}, modality={modality!r} "
                "(conceptual matrix #5/#8/#9/#10)"
            )

    dl = df[df["model"].isin(DL_MODELS)]
    assert len(dl) == UNIFIED_DL_ROWS, (
        f"expected {UNIFIED_DL_ROWS} DL rows (2 models × 2 modalities); got {len(dl)}"
    )

    # The 16 classical rows must survive the DL merge (long-format preservation).
    classical = df[df["model"].isin(CLASSICAL_MODELS)]
    assert len(classical) == UNIFIED_CLASSICAL_ROWS, (
        f"the {UNIFIED_CLASSICAL_ROWS} classical rows must survive the DL merge; "
        f"got {len(classical)}"
    )

    assert len(df) == UNIFIED_TOTAL_ROWS, (
        f"unified_comparison.csv must hold {UNIFIED_TOTAL_ROWS} long-format rows "
        f"(16 classical + 4 DL); got {len(df)} (do NOT collapse to 10 — A1)"
    )
