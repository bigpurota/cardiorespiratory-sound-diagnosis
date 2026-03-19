"""Dataset integrity checks for the downloaded raw audio.

Tests that:
  - CinC 2016 (PhysioNet) has exactly 3,126 WAV files across training-a through -e
    and that each sub-database REFERENCE.csv has the correct row count.
  - ICBHI 2017 has exactly 920 WAV files and 920 matching annotation TXT files.

Each test skips with an explanatory message when the data directories are absent
(i.e., before the download scripts have been run).
"""
import csv
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import (
    CINC_ROOT,
    CINC_DB_COUNTS,
    CINC_EXPECTED,
    ICBHI_ROOT,
    ICBHI_EXPECTED_WAV,
    ICBHI_EXPECTED_TXT,
)


def test_cinc2016_count():
    """Exactly 3,126 WAV recordings in training-A through -E."""
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
    """Each sub-database has a REFERENCE.csv with the expected row count."""
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


def test_icbhi_wav_count():
    """Exactly 920 WAV files in ICBHI 2017."""
    if not ICBHI_ROOT.exists():
        pytest.skip(
            "ICBHI 2017 not downloaded yet — run scripts/download_lung.py first"
        )

    wav_count = sum(1 for _ in ICBHI_ROOT.rglob("*.wav"))
    assert wav_count == ICBHI_EXPECTED_WAV, (
        f"Expected {ICBHI_EXPECTED_WAV} WAV files in {ICBHI_ROOT}, got {wav_count}."
    )


def test_icbhi_annotation_count():
    """Each WAV file in ICBHI 2017 has a matching annotation TXT file."""
    if not ICBHI_ROOT.exists():
        pytest.skip(
            "ICBHI 2017 not downloaded yet — run scripts/download_lung.py first"
        )

    wav_stems = {p.stem for p in ICBHI_ROOT.rglob("*.wav")}
    txt_stems = {p.stem for p in ICBHI_ROOT.rglob("*.txt") if p.stem in wav_stems}

    assert len(txt_stems) == ICBHI_EXPECTED_TXT, (
        f"Expected {ICBHI_EXPECTED_TXT} annotation TXT files (matched to WAVs) "
        f"in {ICBHI_ROOT}, got {len(txt_stems)}. "
        "Verify that the Kaggle download completed and was fully unzipped."
    )
