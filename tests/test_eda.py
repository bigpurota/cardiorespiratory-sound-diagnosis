"""EDA figure assertions for the plots saved under results/figures/eda/.

The EDA scripts save PNG figures covering:
  - class distribution for both modalities (heart, lung)
  - duration histograms for both modalities
  - heart per-DB (A–E) recording counts
  - lung per-class cycle counts
  - ICBHI native sampling-rate distribution

This is a file-existence + non-empty check only (plot content is verified manually).
The test skips when the EDA directory is empty/absent and asserts once the EDA scripts
have run.
"""
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
EDA_DIR = PROJECT_ROOT / "results" / "figures" / "eda"

# Substrings expected in the EDA PNG filenames (one figure per concern).
EXPECTED_FIGURE_KEYWORDS = [
    "class_dist_heart",
    "class_dist_lung",
    "duration_hist_heart",
    "duration_hist_lung",
    "heart_per_db",
    "lung_per_class",
    "icbhi_native_sr",
]


def test_eda_figures_exist():
    """The expected EDA PNGs must exist and be non-empty under results/figures/eda/."""
    if not EDA_DIR.exists() or not any(EDA_DIR.glob("*.png")):
        pytest.skip(
            "results/figures/eda/ has no PNGs yet — run the EDA scripts first"
        )

    pngs = list(EDA_DIR.glob("*.png"))
    names = [p.name.lower() for p in pngs]

    # Every expected concern must be represented by at least one non-empty PNG.
    missing = []
    for keyword in EXPECTED_FIGURE_KEYWORDS:
        matches = [p for p in pngs if keyword in p.name.lower()]
        if not matches:
            missing.append(keyword)
        else:
            for m in matches:
                assert m.stat().st_size > 0, f"EDA figure {m.name} is empty (0 bytes)"

    assert not missing, (
        f"missing EDA figures for: {missing} "
        f"(present under results/figures/eda/: {names})"
    )
