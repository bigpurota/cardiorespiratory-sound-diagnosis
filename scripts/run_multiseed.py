"""
Multi-seed run of the deep-learning configs for mean±std reporting.

Runs all 4 DL configs (heart and lung, each with cnn and effnet_b0) across seeds {42, 1, 2}
as 12 subprocess jobs distributed over up to 4 GPUs via CUDA_VISIBLE_DEVICES. Each job calls
run_modality with the HPO-chosen config (from hpo_best_configs.json, falling back to defaults
with a warning if absent) and a per-seed RNG; results are aggregated into mean±std per row.

After aggregation it idempotently replaces the 4 DL rows in unified_comparison.csv with the
multi-seed mean, leaving the classical rows intact.

Output:
  results/tables/metrics_multiseed.csv     — mean±std per DL row
  results/tables/metrics_multiseed_raw.csv — raw per-seed rows for provenance
  results/tables/unified_comparison.csv    — 4 DL rows updated to multi-seed mean
  results/tables/volumetrics_cnn.csv       — refreshed (params + mean train_time_s)
  results/figures/learning_curve_{heart,lung}_{cnn,effnet}.png (from the seed=42 run)
  results/figures/cm_{heart,lung}_{cnn,effnet}.png             (from the seed=42 run)

Usage:
    OMP_NUM_THREADS=8 .venv/bin/python scripts/run_multiseed.py --wall-cap-min 120
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

FEATURES_DIR = PROJECT_ROOT / "features"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
MULTISEED_CSV = TABLES_DIR / "metrics_multiseed.csv"
HPO_JSON = TABLES_DIR / "hpo_best_configs.json"
UNIFIED_CSV = TABLES_DIR / "unified_comparison.csv"
VOLUMETRICS_CSV = TABLES_DIR / "volumetrics_cnn.csv"

SEEDS = [42, 1, 2]
MODALITIES = ["heart", "lung"]
MODELS = ["cnn", "effnet"]

NUM_GPUS = 4

UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]

VOLUMETRICS_COLUMNS = [
    "modality", "feature_set", "model", "train_time_s",
    "n_train_segments", "n_test_segments",
    "n_train_recordings", "n_test_recordings",
    "n_train_patients", "n_test_patients",
    "params", "fallback_from", "data_volume_mb",
]


def _load_hpo_best_configs():
    """Load HPO-chosen configs from hpo_best_configs.json; warn and return {} if absent.

    Returns a dict keyed ``'{modality}_{model_key}'`` (e.g. ``'heart_cnn'``). Callers fall
    back to default hyperparameters when a key is missing.
    """
    if not HPO_JSON.exists():
        print(f"[multiseed] WARNING: {HPO_JSON} not found — using default hyperparams.")
        return {}
    with open(HPO_JSON) as f:
        best_config = json.load(f)
    print(f"[multiseed] Loaded HPO best configs for: {list(best_config.keys())}")
    return best_config


def _run_one_seed(modality, model, seed, wall_cap_min, gpu_id, python_exe, hparams=None):
    """Run a single-seed experiment in a subprocess; return (row_dict or None, elapsed_s).

    Passes the HPO-chosen ``hparams`` into run_modality so each seed re-runs the chosen
    config; falls back to defaults when ``hparams`` is None.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = "8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if hparams is not None:
        safe_hparams = {
            k: v for k, v in hparams.items()
            if k not in ("best_val_score", "trial_idx") and isinstance(v, (int, float, str, bool, list, type(None)))
        }
        hparams_json = json.dumps(safe_hparams)
    else:
        hparams_json = "{}"

    code = f"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS","8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
sys.path.insert(0, "{PROJECT_ROOT}")
import numpy as np
from src.train_cnn import run_modality
cache_path = "{FEATURES_DIR}/{modality}_spectrograms.npy"
cache = np.load(cache_path, allow_pickle=True).item()
hparams = json.loads({hparams_json!r})
# Use HPO-chosen hyperparameters; fall back to defaults for missing keys.
row = run_modality(
    cache, "{modality}", model="{model}",
    wall_cap_s={int(wall_cap_min*60)},
    seed={seed},
    lr=hparams.get("learning_rate"),
    batch_size=hparams.get("batch_size", 32),
    max_epochs=30,
    patience=hparams.get("patience", 7),
    weight_decay=hparams.get("weight_decay", 0.0),
    label_smoothing=hparams.get("label_smoothing", 0.0),
    aug_strength=hparams.get("aug_strength", 1.0),
    sampler_mode=hparams.get("sampler_mode", "class_weight"),
    cnn_widths=hparams.get("cnn_widths"),
    p=hparams.get("p", 0.3),
)
# Serialize only JSON-safe scalar fields.
out = {{k: v for k, v in row.items() if isinstance(v, (int, float, str, bool, type(None)))}}
print("__ROW__" + json.dumps(out))
"""
    t0 = time.time()
    try:
        result = subprocess.run(
            [python_exe, "-c", code],
            capture_output=True, text=True, env=env,
            timeout=max(300, int(wall_cap_min * 120))
        )
        elapsed = time.time() - t0
        for line in result.stdout.splitlines():
            if line.startswith("__ROW__"):
                row = json.loads(line[len("__ROW__"):])
                return row, elapsed
        print(f"  [seed={seed} {modality}/{model}] no __ROW__ in stdout. stderr tail:")
        for l in result.stderr.splitlines()[-20:]:
            print("   ", l)
        return None, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [seed={seed} {modality}/{model}] TIMEOUT after {elapsed:.0f}s")
        return None, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [seed={seed} {modality}/{model}] ERROR: {e}")
        return None, elapsed


def _stable_figure_names(modality, model_name, row):
    """Move per-experiment PNGs to the canonical flat figure names.

    Canonical names: results/figures/learning_curve_{modality}_{cnn|effnet}.png
                     results/figures/cm_{modality}_{cnn|effnet}.png
    """
    fig_stem = "effnet" if model_name == "effnet_b0" else "cnn"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    src_curve = row.get("learning_curve_png") or row.get("curve_png")
    dst_curve = str(FIGURES_DIR / f"learning_curve_{modality}_{fig_stem}.png")
    if src_curve and os.path.exists(src_curve) and os.path.abspath(src_curve) != os.path.abspath(dst_curve):
        shutil.move(src_curve, dst_curve)
    row["learning_curve_png"] = dst_curve

    src_cm = row.get("cm_figure_path")
    dst_cm = str(FIGURES_DIR / f"cm_{modality}_{fig_stem}.png")
    if src_cm and os.path.exists(src_cm) and os.path.abspath(src_cm) != os.path.abspath(dst_cm):
        shutil.move(src_cm, dst_cm)
    row["cm_figure_path"] = dst_cm
    row["cm_figure"] = os.path.basename(dst_cm)
    return row


def _rebuild_unified(modality, model_name, row):
    """Idempotently merge one DL row into unified_comparison.csv.

    Drops this modality's matching DL row, then appends the new one; the classical rows
    survive intact.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([{c: row.get(c, "") for c in UNIFIED_COLUMNS}])

    if UNIFIED_CSV.exists():
        existing = pd.read_csv(UNIFIED_CSV)
        for c in UNIFIED_COLUMNS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[UNIFIED_COLUMNS]
        drop_mask = (
            existing["model"].isin(["cnn", "effnet_b0"])
            & (existing["modality"] == modality)
            & (existing["model"] == model_name)
        )
        existing = existing[~drop_mask]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined = combined[UNIFIED_COLUMNS]
    combined.to_csv(UNIFIED_CSV, index=False)
    return combined


def _rebuild_volumetrics(modality, model_name, row):
    """Merge one DL volumetrics row into volumetrics_cnn.csv (params + mean train_time_s)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FEATURES_DIR / f"{modality}_spectrograms.npy"
    data_volume_mb = (os.path.getsize(cache_path) / 1e6) if cache_path.exists() else 0.0

    new_row = {c: row.get(c, "") for c in VOLUMETRICS_COLUMNS}
    new_row["data_volume_mb"] = round(data_volume_mb, 3)
    new = pd.DataFrame([new_row])[VOLUMETRICS_COLUMNS]

    if VOLUMETRICS_CSV.exists():
        existing = pd.read_csv(VOLUMETRICS_CSV)
        for c in VOLUMETRICS_COLUMNS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[VOLUMETRICS_COLUMNS]
        existing = existing[~((existing["modality"] == modality) & (existing["model"] == model_name))]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined = combined[VOLUMETRICS_COLUMNS]
    combined.to_csv(VOLUMETRICS_CSV, index=False)
    return combined


def main():
    ap = argparse.ArgumentParser(description="Multi-seed DL run for A-priority enrichment.")
    ap.add_argument("--wall-cap-min", type=float, default=45.0,
                    help="Per-experiment wall-clock cap in minutes.")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS,
                    help="Seeds to run (default: 42 1 2).")
    ap.add_argument("--modalities", nargs="+", default=MODALITIES,
                    choices=["heart", "lung"], help="Modalities to run.")
    ap.add_argument("--models", nargs="+", default=MODELS,
                    choices=["cnn", "effnet"], help="Models to run.")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter path.")
    args = ap.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    python_exe = args.python

    hpo_best_config = _load_hpo_best_configs()

    jobs = list(product(args.modalities, args.models, args.seeds))
    print(f"[multiseed] {len(jobs)} jobs: {args.modalities} × {args.models} × seeds={args.seeds}")
    print(f"[multiseed] GPUs=0-{NUM_GPUS-1}, wall_cap={args.wall_cap_min}min/job")

    all_rows = []
    seed42_rows = {}

    for idx, (modality, model, seed) in enumerate(jobs):
        model_key = "effnet" if model in ("effnet", "effnet_b0") else "cnn"
        config_key = f"{modality}_{model_key}"

        hparams = hpo_best_config.get(config_key)
        if hparams is None:
            print(f"  [multiseed] WARNING: no HPO config for '{config_key}' — using defaults.")

        gpu_id = idx % NUM_GPUS
        print(f"\n[{idx+1}/{len(jobs)}] {modality}/{model} seed={seed} -> GPU {gpu_id}"
              + (f" (HPO: val_score={hparams.get('best_val_score', '?'):.4f})" if hparams else " (defaults)"))

        row, elapsed = _run_one_seed(
            modality, model, seed, args.wall_cap_min, gpu_id, python_exe, hparams=hparams
        )
        if row is not None:
            row["seed"] = seed
            row["gpu_id"] = gpu_id
            row["elapsed_s"] = round(elapsed, 1)
            all_rows.append(row)
            pm = row.get("primary_metric", "?")
            print(f"  -> primary_metric={pm} elapsed={elapsed:.0f}s")

            if seed == 42:
                model_name = row.get("model", model_key)
                seed42_rows[(modality, model_name)] = row
        else:
            print(f"  -> FAILED (elapsed={elapsed:.0f}s)")

    if not all_rows:
        print("[multiseed] All jobs failed — no results to aggregate.")
        sys.exit(1)

    raw_df = pd.DataFrame(all_rows)
    print(f"\n[multiseed] Collected {len(raw_df)} rows from {len(jobs)} jobs.")

    for alias in ("sensitivity", "se"):
        if alias in raw_df.columns and "Se" not in raw_df.columns:
            raw_df["Se"] = raw_df[alias]
    for alias in ("specificity", "sp"):
        if alias in raw_df.columns and "Sp" not in raw_df.columns:
            raw_df["Sp"] = raw_df[alias]

    summary_rows = []
    for (modality, model_name), grp in raw_df.groupby(["modality", "model"]):
        n = len(grp)
        seeds_used = sorted(grp["seed"].tolist())
        pm_name = grp["primary_metric_name"].iloc[0] if "primary_metric_name" in grp.columns else "primary_metric"
        srow = {
            "modality": modality,
            "model": model_name,
            "primary_metric_name": pm_name,
            "mean": round(float(grp["primary_metric"].mean()), 4) if "primary_metric" in grp.columns else None,
            "std": round(float(grp["primary_metric"].std(ddof=1)), 4) if "primary_metric" in grp.columns and n > 1 else 0.0,
            "n_seeds": n,
            "seeds": str(seeds_used),
        }
        for col in ["Se", "Sp", "macro_f1", "auc_roc", "accuracy"]:
            if col in grp.columns:
                srow[f"{col}_mean"] = round(float(pd.to_numeric(grp[col], errors="coerce").mean()), 4)
                srow[f"{col}_std"] = round(float(pd.to_numeric(grp[col], errors="coerce").std(ddof=1)), 4) if n > 1 else 0.0
            else:
                srow[f"{col}_mean"] = None
                srow[f"{col}_std"] = None
        summary_rows.append(srow)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(MULTISEED_CSV, index=False)
    print(f"\n[wrote] {MULTISEED_CSV} ({len(summary_df)} rows)")

    raw_out = TABLES_DIR / "metrics_multiseed_raw.csv"
    safe_cols = [c for c in raw_df.columns if raw_df[c].dtype != object
                 or raw_df[c].apply(lambda x: isinstance(x, (str, type(None)))).all()]
    raw_df[safe_cols].to_csv(raw_out, index=False)
    print(f"[wrote] {raw_out} ({len(raw_df)} raw rows)")

    print("\n[multiseed] Updating unified_comparison.csv (DL rows → multi-seed mean) ...")
    for _, srow in summary_df.iterrows():
        modality = srow["modality"]
        model_name = srow["model"]

        rep_rows = raw_df[(raw_df["modality"] == modality) & (raw_df["model"] == model_name)]
        rep = rep_rows.iloc[0] if len(rep_rows) > 0 else {}

        unified_row = {
            "modality": modality,
            "feature_set": rep.get("feature_set", "log_mel_64x128") if isinstance(rep, pd.Series) else "log_mel_64x128",
            "model": model_name,
            "primary_metric_name": srow.get("primary_metric_name", "primary_metric"),
            "primary_metric": srow["mean"],
            "Se": srow.get("Se_mean", ""),
            "Sp": srow.get("Sp_mean", ""),
            "macro_f1": srow.get("macro_f1_mean", ""),
            "auc_roc": srow.get("auc_roc_mean", ""),
            "accuracy": srow.get("accuracy_mean", ""),
            "n_train": rep.get("n_train", "") if isinstance(rep, pd.Series) else "",
            "n_test": rep.get("n_test", "") if isinstance(rep, pd.Series) else "",
        }
        _rebuild_unified(modality, model_name, unified_row)

    if UNIFIED_CSV.exists():
        df_u = pd.read_csv(UNIFIED_CSV)
        n_rows = len(df_u)
        model_tags = set(df_u["model"].tolist())
        print(f"[multiseed] unified_comparison.csv: {n_rows} rows, models={sorted(model_tags)}")
        if n_rows != 20:
            print(f"  WARNING: expected 20 rows, got {n_rows}")

    print("\n[multiseed] Refreshing figures from seed=42 runs ...")
    for (modality, model_name), row in seed42_rows.items():
        row = _stable_figure_names(modality, model_name, row)
        print(f"  [{modality}/{model_name}] learning_curve -> {row.get('learning_curve_png')}")
        print(f"  [{modality}/{model_name}] cm -> {row.get('cm_figure_path')}")

    print("\n[multiseed] Updating volumetrics_cnn.csv ...")
    for (modality, model_name), grp in raw_df.groupby(["modality", "model"]):
        rep = grp.iloc[0]
        vrow = dict(rep)
        vrow["train_time_s"] = round(float(grp["train_time_s"].mean()), 1) if "train_time_s" in grp.columns else ""
        vrow["fallback_from"] = rep.get("fallback_from", "")
        _rebuild_volumetrics(modality, model_name, vrow)

    print(f"[wrote] {VOLUMETRICS_CSV}")

    print("\n=== MULTI-SEED SUMMARY ===")
    for _, r in summary_df.iterrows():
        print(f"  {r['modality']:6s} {r['model']:10s}  "
              f"{r['primary_metric_name']}={r['mean']:.4f}±{r['std']:.4f}  "
              f"n_seeds={r['n_seeds']}  seeds={r['seeds']}")
    print("=========================")
    print("Done.")


if __name__ == "__main__":
    main()
