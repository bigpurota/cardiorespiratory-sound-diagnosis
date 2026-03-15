"""
tests/test_data_integrity.py — DATA-01 and DATA-02 assertions.

Tests that:
  - CinC 2016 (PhysioNet) has exactly 3,126 WAV files across training-a through -e
    and that each sub-database REFERENCE.csv has the correct row count (DATA-01).
  - ICBHI 2017 has exactly 920 WAV files and 920 matching annotation TXT files (DATA-02).

Wave 0: All four tests will SKIP with explanatory messages when the data directories
are absent (i.e., before the download scripts have been run). Collection must succeed
with 0 errors.
"""
import csv
import pathlib
import sys

import pytest

# Make conftest constants available regardless of how pytest resolves the package.
# When tests/ is a package and conftest.py lives in it, pytest injects conftest
# into sys.modules under the key "conftest" at collection time. The sys.path
# insertion below ensures the import also works when running pytest directly from
# the project root with python3 -m pytest.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import (  # noqa: E501
    CINC_ROOT,
    CINC_DB_COUNTS,
    CINC_EXPECTED,
    ICBHI_ROOT,
    ICBHI_EXPECTED_WAV,
    ICBHI_EXPECTED_TXT,
)


# ---------------------------------------------------------------------------
# CinC 2016 — heart sound (DATA-01)
# ---------------------------------------------------------------------------

def test_cinc2016_count():
    """ROADMAP success criterion 1: exactly 3,126 WAV recordings in training-A through -E."""
    if not CINC_ROOT.exists():
        pytest.skip(
            "CinC 2016 not downloaded yet — run scripts/download_heart.py first"
        )

    wav_count = sum(1 for _ in CINC_ROOT.rglob("*.wav"))
    assert wav_count == CINC_EXPECTED, (
        f"Expected {CINC_EXPECTED} WAV files in {CINC_ROOT}, got {wav_count}. "
        "Make sure training-f is NOT included (A–E only = 3,126; A–F = 3,240)."
    )


def test_cinc2016_references():
    """Each sub-database has a REFERENCE.csv with the expected row count — DATA-01."""
    if not CINC_ROOT.exists():
        pytest.skip(
            "CinC 2016 not downloaded yet — run scripts/download_heart.py first"
        )

    for db, expected_count in CINC_DB_COUNTS.items():
        ref_path = CINC_ROOT / db / "REFERENCE.csv"
        assert ref_path.exists(), (
            f"REFERENCE.csv missing for {db} — expected at {ref_path}"
        )
        with open(ref_path, newline="") as fh:
            rows = list(csv.reader(fh))
        assert len(rows) == expected_count, (
            f"{db}/REFERENCE.csv: expected {expected_count} rows, got {len(rows)}"
        )


# ---------------------------------------------------------------------------
# ICBHI 2017 — lung sound (DATA-02)
# ---------------------------------------------------------------------------

def test_icbhi_wav_count():
    """ROADMAP success criterion 2: exactly 920 WAV files in ICBHI 2017."""
    if not ICBHI_ROOT.exists():
        pytest.skip(
            "ICBHI 2017 not downloaded yet — run scripts/download_lung.py first"
        )

    wav_count = sum(1 for _ in ICBHI_ROOT.rglob("*.wav"))
    assert wav_count == ICBHI_EXPECTED_WAV, (
        f"Expected {ICBHI_EXPECTED_WAV} WAV files in {ICBHI_ROOT}, got {wav_count}."
    )


def test_icbhi_annotation_count():
    """Each WAV file in ICBHI 2017 has a matching annotation TXT file — DATA-02."""
    if not ICBHI_ROOT.exists():
        pytest.skip(
            "ICBHI 2017 not downloaded yet — run scripts/download_lung.py first"
        )

    wav_stems = {p.stem for p in ICBHI_ROOT.rglob("*.wav")}
    # Only count TXT files whose stem matches a WAV file (exclude metadata TXTs)
    txt_stems = {p.stem for p in ICBHI_ROOT.rglob("*.txt") if p.stem in wav_stems}

    assert len(txt_stems) == ICBHI_EXPECTED_TXT, (
        f"Expected {ICBHI_EXPECTED_TXT} annotation TXT files (matched to WAVs) "
        f"in {ICBHI_ROOT}, got {len(txt_stems)}. "
        "Verify that the Kaggle download completed and was fully unzipped."
    )
