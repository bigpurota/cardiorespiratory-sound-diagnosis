"""Manifest checks for the ingest step (src/ingest.py).

Covers the schema and label distributions of data/processed/manifest.csv and
data/processed/lung_cycles.csv. Every test skips when the artifacts are not
present, so collection works on a fresh checkout.
"""
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).parent))

MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
LUNG_CYCLES_PATH = PROJECT_ROOT / "data" / "processed" / "lung_cycles.csv"

# Exact, order-sensitive manifest header.
EXPECTED_COLUMNS = [
    "filepath",
    "patient_id",
    "label",
    "modality",
    "duration_s",
    "db_source",
    "segment_id",
]

# Heart label distribution across databases A–E.
HEART_NORMAL_COUNT = 2495    # label == -1
HEART_ABNORMAL_COUNT = 631   # label == 1

# ICBHI cycle-level 4-class distribution.
LUNG_CLASS_COUNTS = {
    "normal": 3642,
    "crackle": 1864,
    "wheeze": 886,
    "both": 506,
}


def _read_csv(path):
    """Read a CSV with pandas, skipping the test if the file is absent."""
    if not path.exists():
        pytest.skip(f"{path.name} not built yet — run the ingest step first")
    import pandas as pd
    return pd.read_csv(path)


# Schema + total join

def test_columns():
    """manifest.csv header must equal the expected columns, in order."""
    df = _read_csv(MANIFEST_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"manifest columns must be exactly {EXPECTED_COLUMNS} (order-sensitive); "
        f"got {list(df.columns)}"
    )


def test_files_exist():
    """Every `filepath` in the manifest must point to an existing file (total join)."""
    df = _read_csv(MANIFEST_PATH)
    missing = []
    for raw in df["filepath"].astype(str):
        p = pathlib.Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            missing.append(raw)
    assert not missing, (
        f"{len(missing)} manifest filepaths do not exist on disk, e.g. {missing[:5]}"
    )


# Label distributions

def test_heart_label_dist():
    """Heart rows: 2495 normal(-1) / 631 abnormal(1); zero unsure(0) rows."""
    df = _read_csv(MANIFEST_PATH)
    heart = df[df["modality"] == "heart"]
    counts = heart["label"].value_counts().to_dict()

    n_normal = counts.get(-1, 0)
    n_abnormal = counts.get(1, 0)
    n_unsure = counts.get(0, 0)

    assert n_normal == HEART_NORMAL_COUNT, (
        f"heart normal(-1) count must be {HEART_NORMAL_COUNT}, got {n_normal}"
    )
    assert n_abnormal == HEART_ABNORMAL_COUNT, (
        f"heart abnormal(1) count must be {HEART_ABNORMAL_COUNT}, got {n_abnormal}"
    )
    assert n_unsure == 0, (
        f"heart unsure(0) rows must be excluded (count 0), got {n_unsure}"
    )


def test_lung_label_dist():
    """Lung cycles: 4-class distribution normal:3642 crackle:1864 wheeze:886 both:506."""
    df = _read_csv(LUNG_CYCLES_PATH)
    counts = df["label"].astype(str).str.lower().value_counts().to_dict()
    for cls, expected in LUNG_CLASS_COUNTS.items():
        got = counts.get(cls, 0)
        assert got == expected, (
            f"lung class '{cls}' count must be {expected}, got {got}"
        )
