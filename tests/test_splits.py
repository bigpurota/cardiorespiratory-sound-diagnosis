"""
tests/test_splits.py — DATA-03 patient-level split assertions (Wave 0, RED).

Specifies the contracts for the Wave-2 split builder (src/split.py) per
02-RESEARCH.md Code Examples §4/§5/§8 and the ROADMAP isdisjoint criterion:

  - src.split.assert_no_patient_leakage(train_ids, test_ids) — importable helper
    that RAISES AssertionError on any patient overlap and returns/logs cleanly on
    disjoint sets (the reusable zero-leakage guard imported by every Phase-3 script).
  - results/splits/heart_splits.csv and results/splits/lung_splits.csv each satisfy
    set(train patient_ids).isdisjoint(set(test patient_ids)).
  - each split CSV encodes a `patient_id` column and a train/test `split` flag;
    the heart split additionally carries `db_source` (A–E provenance).
  - the lung split records provenance (official-with-repair vs reconstructed); if
    the official path was taken, the overlap patients 156 and 218 are repaired so
    each appears on exactly ONE side.

The helper test imports src.split and SKIPS if it is absent (Wave 0). The
artifact tests SKIP when the split CSVs / provenance record are absent. Collection
MUST succeed with zero errors.
"""
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity

SPLITS_DIR = PROJECT_ROOT / "results" / "splits"
HEART_SPLITS = SPLITS_DIR / "heart_splits.csv"
LUNG_SPLITS = SPLITS_DIR / "lung_splits.csv"
LUNG_PROVENANCE = SPLITS_DIR / "lung_split_provenance.txt"

# VERIFIED: the official ICBHI split has these 2 patients in BOTH train and test.
OVERLAP_PATIENTS = {"156", "218"}


def _import_split():
    """Import src.split, skipping (not erroring) if absent in Wave 0."""
    try:
        import src.split as split_mod
    except Exception as exc:  # pragma: no cover - defensive (Wave 2 module absent)
        pytest.skip(f"src.split not implemented yet (Wave 0): {exc}")
    return split_mod


def _read_split(path):
    """Read a split CSV with pandas, skipping if absent (Wave 0)."""
    if not path.exists():
        pytest.skip(f"{path.name} not built yet — run the Wave-2 split step first")
    import pandas as pd
    return pd.read_csv(path)


def _split_column(df):
    """Return the name of the train/test flag column (`split`)."""
    for cand in ("split", "fold", "set"):
        if cand in df.columns:
            return cand
    raise AssertionError(f"no train/test split flag column found in {list(df.columns)}")


# ---------------------------------------------------------------------------
# Reusable zero-leakage helper
# ---------------------------------------------------------------------------

def test_leakage_helper():
    """assert_no_patient_leakage raises on overlap and passes cleanly on disjoint sets."""
    split_mod = _import_split()
    assert hasattr(split_mod, "assert_no_patient_leakage"), (
        "src.split must export an importable assert_no_patient_leakage helper"
    )
    fn = split_mod.assert_no_patient_leakage

    # Disjoint sets: must NOT raise.
    fn(["p1", "p2", "p3"], ["p4", "p5"])

    # Overlapping sets: MUST raise AssertionError.
    with pytest.raises(AssertionError):
        fn(["p1", "p2", "p3"], ["p3", "p4"])


# ---------------------------------------------------------------------------
# Disjointness of the on-disk splits
# ---------------------------------------------------------------------------

def test_splits_disjoint():
    """Both heart and lung splits must have train/test patient sets that are disjoint."""
    for path in (HEART_SPLITS, LUNG_SPLITS):
        df = _read_split(path)
        flag = _split_column(df)
        train_ids = set(df[df[flag] == "train"]["patient_id"].astype(str))
        test_ids = set(df[df[flag] == "test"]["patient_id"].astype(str))
        assert train_ids.isdisjoint(test_ids), (
            f"{path.name}: train/test patient sets overlap "
            f"(e.g. {sorted(train_ids & test_ids)[:5]})"
        )


def test_split_schema():
    """Each split CSV encodes patient_id + a train/test flag; heart also carries db_source."""
    heart = _read_split(HEART_SPLITS)
    assert "patient_id" in heart.columns, "heart split missing patient_id column"
    _split_column(heart)  # raises if no train/test flag
    assert "db_source" in heart.columns, (
        "heart split must carry db_source (A–E provenance, D-10)"
    )

    lung = _read_split(LUNG_SPLITS)
    assert "patient_id" in lung.columns, "lung split missing patient_id column"
    _split_column(lung)  # raises if no train/test flag


# ---------------------------------------------------------------------------
# Lung split provenance + overlap repair (patients 156, 218)
# ---------------------------------------------------------------------------

def test_lung_split_provenance():
    """A provenance record states the path taken; if official, 156/218 are repaired.

    The lung split build path is one of {official-with-repair, reconstructed}.
    A provenance record (results/splits/lung_split_provenance.txt) must state which
    path was taken (D-04). When the official path is taken, the known overlap
    patients 156 and 218 must each land on exactly ONE side (repair applied).
    """
    if not LUNG_PROVENANCE.exists():
        pytest.skip(
            "lung_split_provenance.txt not written yet — run the Wave-2 split step first"
        )
    provenance = LUNG_PROVENANCE.read_text(encoding="utf-8").lower()
    assert ("official" in provenance) or ("reconstruct" in provenance), (
        "provenance record must state which split path was taken "
        "(official-with-repair vs reconstructed)"
    )

    if "official" in provenance:
        df = _read_split(LUNG_SPLITS)
        flag = _split_column(df)
        for pid in OVERLAP_PATIENTS:
            sides = set(df[df["patient_id"].astype(str) == pid][flag].astype(str))
            assert len(sides) <= 1, (
                f"overlap patient {pid} must be repaired to exactly one side, "
                f"found on {sorted(sides)}"
            )
