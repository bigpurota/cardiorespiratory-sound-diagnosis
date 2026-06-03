"""Tests for the patient-level split builder (src/split.py)."""
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

SPLITS_DIR = PROJECT_ROOT / "results" / "splits"
HEART_SPLITS = SPLITS_DIR / "heart_splits.csv"
LUNG_SPLITS = SPLITS_DIR / "lung_splits.csv"
LUNG_PROVENANCE = SPLITS_DIR / "lung_split_provenance.txt"

OVERLAP_PATIENTS = {"156", "218"}


def _import_split():
    """Import src.split, skipping (not erroring) if absent."""
    try:
        import src.split as split_mod
    except Exception as exc:
        pytest.skip(f"src.split not implemented yet: {exc}")
    return split_mod


def _read_split(path):
    """Read a split CSV with pandas, skipping if absent."""
    if not path.exists():
        pytest.skip(f"{path.name} not built yet — run the split step first")
    import pandas as pd
    return pd.read_csv(path)


def _split_column(df):
    """Return the name of the train/test flag column (`split`)."""
    for cand in ("split", "fold", "set"):
        if cand in df.columns:
            return cand
    raise AssertionError(f"no train/test split flag column found in {list(df.columns)}")


def test_leakage_helper():
    """assert_no_patient_leakage raises on overlap and passes"""
    split_mod = _import_split()
    assert hasattr(split_mod, "assert_no_patient_leakage"), (
        "src.split must export an importable assert_no_patient_leakage helper"
    )
    fn = split_mod.assert_no_patient_leakage

    fn(["p1", "p2", "p3"], ["p4", "p5"])

    with pytest.raises(AssertionError):
        fn(["p1", "p2", "p3"], ["p3", "p4"])


def test_splits_disjoint():
    """Both heart and lung splits must have train/test patient"""
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
    """Each split CSV encodes patient_id + a train/test flag;"""
    heart = _read_split(HEART_SPLITS)
    assert "patient_id" in heart.columns, "heart split missing patient_id column"
    _split_column(heart)
    assert "db_source" in heart.columns, (
        "heart split must carry db_source (A–E provenance)"
    )

    lung = _read_split(LUNG_SPLITS)
    assert "patient_id" in lung.columns, "lung split missing patient_id column"
    _split_column(lung)


def test_lung_split_provenance():
    """A provenance record states the path taken; if official,"""
    if not LUNG_PROVENANCE.exists():
        pytest.skip(
            "lung_split_provenance.txt not written yet — run the split step first"
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
