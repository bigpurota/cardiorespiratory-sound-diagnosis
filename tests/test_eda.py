"""EDA figure assertions for the plots saved under"""
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
EDA_DIR = PROJECT_ROOT / "results" / "figures" / "eda"

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
    """The expected EDA PNGs must exist and be non-empty under"""
    if not EDA_DIR.exists() or not any(EDA_DIR.glob("*.png")):
        pytest.skip(
            "results/figures/eda/ has no PNGs yet — run the EDA scripts first"
        )

    pngs = list(EDA_DIR.glob("*.png"))
    names = [p.name.lower() for p in pngs]

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
