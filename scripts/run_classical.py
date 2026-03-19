"""
Run the classical (features + classifiers) experiments per modality and write the CSVs.

CLI ``--modality {heart,lung,all}`` that:
  1. loads the feature cache ``features/{modality}_classical.npy``;
  2. re-asserts ``assert_no_patient_leakage`` on the cached train/test patient groups;
  3. calls ``src.train_classical.run_experiments(modality, cache)`` to fit and evaluate the
     (feature_set × model) experiments, saving a confusion-matrix figure per model;
  4. writes ``results/tables/metrics_{modality}_classical.csv`` (full metric suite headlined
     on MAcc / ICBHI_Score, not accuracy), rebuilds the classical rows of
     ``results/tables/unified_comparison.csv``, and refreshes
     ``results/tables/volumetrics_classical.csv``.

Each modality is independently runnable so a lung failure never loses heart results.

    uv run python scripts/run_classical.py --modality heart
    uv run python scripts/run_classical.py --modality lung
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

from src.split import assert_no_patient_leakage
from src.train_classical import run_experiments, MODEL_NAMES

FEATURES_DIR = os.path.join(config.PROJECT_ROOT, "features")
TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")

UNIFIED_CSV = os.path.join(TABLES_DIR, "unified_comparison.csv")
VOLUMETRICS_CSV = os.path.join(TABLES_DIR, "volumetrics_classical.csv")

UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]

VOLUMETRICS_COLUMNS = [
    "modality", "feature_set", "model", "train_time_s",
    "n_train_segments", "n_test_segments",
    "n_train_recordings", "n_test_recordings",
    "n_train_patients", "n_test_patients",
    "data_volume_mb",
]

METRICS_COLUMNS = {
    "heart": [
        "feature_set", "model", "primary_metric_name", "primary_metric",
        "MAcc", "Se", "Sp", "macro_f1", "auc_roc", "accuracy",
        "n_train", "n_test", "best_params", "cm_figure",
    ],
    "lung": [
        "feature_set", "model", "primary_metric_name", "primary_metric",
        "ICBHI_Score", "Se", "Sp", "macro_f1", "accuracy",
        "se_crackle", "se_wheeze", "se_both", "se_normal",
        "n_train", "n_test", "best_params", "cm_figure",
    ],
}


def _load_cache(modality):
    """Load the feature cache for ``modality`` and return the payload dict + path."""
    path = os.path.join(FEATURES_DIR, f"{modality}_classical.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"feature cache missing: {path} — run "
            f"`uv run python scripts/build_features.py --modality {modality}` first."
        )
    return np.load(path, allow_pickle=True).item(), path


def _write_metrics_csv(modality, rows):
    """Write the per-modality metrics CSV with the full metric suite."""
    cols = METRICS_COLUMNS[modality]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    out = os.path.join(TABLES_DIR, f"metrics_{modality}_classical.csv")
    df.to_csv(out, index=False)
    print(f"[wrote] {out} ({len(df)} rows)")
    return out


def _rebuild_unified(modality, rows):
    """Idempotently merge this modality's classical rows into unified_comparison.csv.

    Drops this modality's existing classical rows, appends the fresh ones, and rewrites in
    column order. Any DL rows written by the other drivers are preserved untouched.
    """
    new = pd.DataFrame([{c: r.get(c, "") for c in UNIFIED_COLUMNS} for r in rows])

    if os.path.exists(UNIFIED_CSV):
        existing = pd.read_csv(UNIFIED_CSV)
        for c in UNIFIED_COLUMNS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[UNIFIED_COLUMNS]
        drop_mask = (
            (existing["modality"] == modality)
            & (existing["model"].isin(MODEL_NAMES))
        )
        existing = existing[~drop_mask]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined = combined[UNIFIED_COLUMNS]
    combined.to_csv(UNIFIED_CSV, index=False)
    n_classical = int(combined["model"].isin(MODEL_NAMES).sum())
    print(f"[wrote] {UNIFIED_CSV} ({len(combined)} rows; {n_classical} classical)")
    return UNIFIED_CSV


def _rebuild_volumetrics(modality, rows, cache_path):
    """Merge this modality's per-run volumetrics into volumetrics_classical.csv."""
    data_volume_mb = os.path.getsize(cache_path) / 1e6
    new_rows = []
    for r in rows:
        vr = {c: r.get(c, "") for c in VOLUMETRICS_COLUMNS}
        vr["data_volume_mb"] = round(data_volume_mb, 3)
        new_rows.append(vr)
    new = pd.DataFrame(new_rows)[VOLUMETRICS_COLUMNS]

    if os.path.exists(VOLUMETRICS_CSV):
        existing = pd.read_csv(VOLUMETRICS_CSV)
        for c in VOLUMETRICS_COLUMNS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[VOLUMETRICS_COLUMNS]
        existing = existing[existing["modality"] != modality]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined = combined[VOLUMETRICS_COLUMNS]
    combined.to_csv(VOLUMETRICS_CSV, index=False)
    print(f"[wrote] {VOLUMETRICS_CSV} ({len(combined)} rows)")
    return VOLUMETRICS_CSV


def run_modality(modality):
    """Run all experiments for ``modality`` and write the three CSVs + CM figures."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    cache, cache_path = _load_cache(modality)

    pid = np.asarray(list(map(str, cache["patient_id"])), dtype=object)
    split = np.asarray(cache["split"], dtype=object)
    assert_no_patient_leakage(pid[split == "train"], pid[split == "test"])

    print(f"[run_classical] modality={modality} cache={cache_path}")
    rows = run_experiments(modality, cache)

    _write_metrics_csv(modality, rows)
    _rebuild_unified(modality, rows)
    _rebuild_volumetrics(modality, rows, cache_path)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Run classical experiments (8 per modality) and write result CSVs."
    )
    ap.add_argument("--modality", required=True, choices=["heart", "lung", "all"])
    args = ap.parse_args()

    modalities = ["heart", "lung"] if args.modality == "all" else [args.modality]
    for m in modalities:
        run_modality(m)
    sys.exit(0)


if __name__ == "__main__":
    main()
