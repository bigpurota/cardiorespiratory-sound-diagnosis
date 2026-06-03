"""Fine-tune the AST over three seeds per modality and aggregate mean +/- std.

Produces the two tables reported for the Audio Spectrogram Transformer in Chapter 4:
  results/tables/ast_multiseed_raw.csv  one row per (modality, seed)
  results/tables/ast_multiseed.csv      mean +/- std per modality

It reuses ``scripts/run_ast._run_ast_modality`` so the per-run logic is identical to
the single-seed driver; only the seed loop, per-seed figure isolation and the
aggregation are added here.

REPRODUCIBILITY NOTE. Each seed reseeds random / numpy / torch / cuda, exactly like the
other deep runs. Unlike the classical and HPO-pinned runs, however, AST training stops
on a WALL-CLOCK budget (``--wall-cap-min``, default 25) rather than a fixed epoch count,
so the number of optimiser steps -- and therefore the exact score -- depends on the
host's throughput. The committed CSVs were produced on a 4x A100 box at 25 min/seed; a
rerun on different hardware lands near, but not bit-identical to, the reported means.
This caveat is stated in the report's Reproducibility section. For a hardware-independent
rerun, edit ``scripts/run_ast._run_ast_modality`` to pass a fixed ``max_epochs`` and a
large ``--wall-cap-min`` so the timer never fires.
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

import numpy as np
import pandas as pd

import scripts.run_ast as ra

SEEDS = (1, 2, 42)
MODALITIES = ("heart", "lung")
TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")
RAW_CSV = os.path.join(TABLES_DIR, "ast_multiseed_raw.csv")
AGG_CSV = os.path.join(TABLES_DIR, "ast_multiseed.csv")


def _one_run(modality, seed, wall_cap_s):
    """Fine-tune AST for one (modality, seed) and return a compact result row."""
    ra.FIGURES_DIR = os.path.join(
        config.RESULTS_DIR, "figures", "ast_multiseed", f"{modality}_{seed}"
    )
    os.makedirs(ra.FIGURES_DIR, exist_ok=True)
    cache, _ = ra._load_cache(modality)
    row = ra._run_ast_modality(
        cache, modality, lr=ra._DEFAULT_LR, wall_cap_s=wall_cap_s, seed=seed
    )
    auc = row.get("auc_roc", "")
    return {
        "modality": modality,
        "seed": seed,
        "primary": round(float(row["primary_metric"]), 4),
        "Se": round(float(row["Se"]), 4),
        "Sp": round(float(row["Sp"]), 4),
        "auc": round(float(auc), 4) if auc not in ("", None) else "",
        "epochs": int(row.get("epochs_ran", 0)),
    }


def _aggregate(new_rows):
    """Merge new rows with any existing raw CSV and rewrite both tables."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    df = pd.DataFrame(new_rows)
    if os.path.exists(RAW_CSV):
        df = pd.concat([pd.read_csv(RAW_CSV), df], ignore_index=True)
    df = df.drop_duplicates(["modality", "seed"], keep="last").sort_values(
        ["modality", "seed"]
    )
    df.to_csv(RAW_CSV, index=False)

    out = []
    for m in MODALITIES:
        sub = df[df["modality"] == m]
        if sub.empty:
            continue
        out.append(
            {
                "modality": m,
                "metric": "MAcc" if m == "heart" else "ICBHI",
                "mean": round(sub["primary"].mean(), 4),
                "std": round(sub["primary"].std(ddof=1), 4),
                "Se_mean": round(sub["Se"].mean(), 4),
                "Sp_mean": round(sub["Sp"].mean(), 4),
                "per_seed": ";".join(f"{v:.4f}" for v in sub["primary"]),
            }
        )
    pd.DataFrame(out).to_csv(AGG_CSV, index=False)
    print(f"[wrote] {RAW_CSV} ({len(df)} rows)")
    print(f"[wrote] {AGG_CSV} ({len(out)} rows)")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune AST over seeds {1,2,42} per modality; aggregate mean+/-std."
    )
    ap.add_argument("--modality", default="all", choices=["heart", "lung", "all"])
    ap.add_argument("--seeds", default="1,2,42", help="Comma-separated seeds.")
    ap.add_argument("--wall-cap-min", type=float, default=25.0)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    mods = list(MODALITIES) if args.modality == "all" else [args.modality]
    wall_cap_s = int(args.wall_cap_min * 60)

    rows = []
    for m in mods:
        for s in seeds:
            print(f"=== AST {m} seed {s} (wall_cap={wall_cap_s}s) ===")
            rows.append(_one_run(m, s, wall_cap_s))
    _aggregate(rows)
    sys.exit(0)


if __name__ == "__main__":
    main()
