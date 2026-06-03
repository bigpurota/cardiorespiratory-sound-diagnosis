"""Render the cross-modal transfer heat-map from the multiseed summary.

Reads results/tables/cross_modal_multiseed.csv and draws the EfficientNet-B0
source x target primary-metric matrix (three-seed means).
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = PROJECT_ROOT / "results" / "tables" / "cross_modal_multiseed.csv"
OUT = PROJECT_ROOT / "results" / "figures" / "cross_modal_heatmap.png"
MODEL = "effnet_b0"


def _val(df, setting, src, tgt):
    row = df[(df.setting == setting) & (df.source_modality == src)
             & (df.target_modality == tgt) & (df.model == MODEL)]
    return float(row["primary_metric_mean"].iloc[0]) if len(row) else float("nan")


def main():
    df = pd.read_csv(CSV)
    sources = ["heart", "lung", "heart+lung"]
    targets = ["heart", "lung"]
    matrix = np.array([
        [_val(df, "in_domain", "heart", "heart"), _val(df, "transfer", "heart", "lung")],
        [_val(df, "transfer", "lung", "heart"), _val(df, "in_domain", "lung", "lung")],
        [_val(df, "joint", "heart+lung", "heart"), _val(df, "joint", "heart+lung", "lung")],
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=11)
    ax.set_xlabel("Target (fine-tune + evaluation) modality", fontsize=11)
    ax.set_ylabel("Source (pre-training) modality", fontsize=11)
    ax.set_title("Cross-Modal Transfer Matrix — EfficientNet-B0 (3-seed mean)", fontsize=11)
    for i in range(len(sources)):
        for j in range(len(targets)):
            v = matrix[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="black" if 0.3 < v < 0.8 else "white")
    plt.colorbar(im, ax=ax, label="Primary metric (MAcc / ICBHI Score)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"[wrote] {OUT}")
    print(matrix)


if __name__ == "__main__":
    main()
