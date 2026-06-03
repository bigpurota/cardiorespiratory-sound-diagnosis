"""Bounded random hyperparameter search, selecting strictly"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import concurrent.futures
import numpy as np
import pandas as pd

FEATURES_DIR = PROJECT_ROOT / "features"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
HPO_CSV = TABLES_DIR / "hpo_results.csv"
HPO_JSON = TABLES_DIR / "hpo_best_configs.json"

NUM_GPUS = 4

SEARCH_SPACE = {
    "batch_size": [16, 32, 64],
    "aug_strength": [0.5, 1.0, 1.5],
    "label_smoothing": [0.0, 0.05, 0.1],
    "imbalance": ["class_weight", "weighted_sampler"],
    "patience": [5, 7, 10],
    "cnn_widths": [(16, 32, 64, 128), (32, 64, 128, 256)],
    "p": [0.3, 0.5],
}

LR_BOUNDS = {
    "cnn":   (3e-4, 3e-3),
    "effnet": (3e-5, 3e-4),
}

WEIGHT_DECAY_INCLUDE_ZERO_PROB = 0.2
WEIGHT_DECAY_LOG_BOUNDS = (1e-6, 1e-3)


def _sample_config(model_key, rng):
    """Sample one hyperparameter config for ``model_key`` from"""
    config = {}
    config["batch_size"] = int(rng.choice(SEARCH_SPACE["batch_size"]))
    config["aug_strength"] = float(rng.choice(SEARCH_SPACE["aug_strength"]))
    config["label_smoothing"] = float(rng.choice(SEARCH_SPACE["label_smoothing"]))
    config["sampler_mode"] = str(rng.choice(SEARCH_SPACE["imbalance"]))
    config["patience"] = int(rng.choice(SEARCH_SPACE["patience"]))

    lo, hi = LR_BOUNDS[model_key]
    config["lr"] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    if rng.uniform() < WEIGHT_DECAY_INCLUDE_ZERO_PROB:
        config["weight_decay"] = 0.0
    else:
        lo_wd, hi_wd = WEIGHT_DECAY_LOG_BOUNDS
        config["weight_decay"] = float(np.exp(rng.uniform(np.log(lo_wd), np.log(hi_wd))))

    if model_key == "cnn":
        idx = int(rng.choice(len(SEARCH_SPACE["cnn_widths"])))
        config["cnn_widths"] = list(SEARCH_SPACE["cnn_widths"][idx])
        config["p"] = float(rng.choice(SEARCH_SPACE["p"]))
    else:
        config["cnn_widths"] = None
        config["p"] = 0.3

    return config


def _run_one_trial(modality, model, trial_idx, hparam_config, wall_cap_min, gpu_id, python_exe, search_seed=42):
    """Run a single HPO trial in a subprocess; return (row_dict"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = "8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cfg_json = json.dumps(hparam_config)
    code = f"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS","8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
sys.path.insert(0, "{PROJECT_ROOT}")
import numpy as np
from src.train_cnn import run_modality
cache_path = "{FEATURES_DIR}/{modality}_spectrograms.npy"
cache = np.load(cache_path, allow_pickle=True).item()
hparams = json.loads({cfg_json!r})
row = run_modality(
    cache, "{modality}", model="{model}",
    wall_cap_s={int(wall_cap_min*60)},
    seed={search_seed},
    lr=hparams.get("lr"),
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
        print(f"  [trial={trial_idx} {modality}/{model}] no __ROW__ in stdout. stderr tail:")
        for l in result.stderr.splitlines()[-20:]:
            print("   ", l)
        return None, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [trial={trial_idx} {modality}/{model}] TIMEOUT after {elapsed:.0f}s")
        return None, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [trial={trial_idx} {modality}/{model}] ERROR: {e}")
        return None, elapsed


def main():
    ap = argparse.ArgumentParser(
        description="Bounded random HPO for SmallCNN + EfficientNet-B0 on heart + lung."
    )
    ap.add_argument("--n-trials", type=int, default=16,
                    help="Number of random trials per (modality, model) (default: 16).")
    ap.add_argument("--wall-cap-min", type=float, default=20.0,
                    help="Per-trial wall-clock cap in minutes (default: 20, relax on GPU).")
    ap.add_argument("--modalities", nargs="+", default=["heart", "lung"],
                    choices=["heart", "lung"])
    ap.add_argument("--models", nargs="+", default=["cnn", "effnet"],
                    choices=["cnn", "effnet"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Search seed(s) — each seed offsets the per-trial RNG (default: 42).")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    python_exe = args.python
    n_trials = args.n_trials

    configs_to_run = list(product(args.modalities, args.models))
    search_seed = args.seeds[0]

    print(f"[hpo] {len(configs_to_run)} configs × {n_trials} trials, "
          f"wall_cap={args.wall_cap_min}min/trial, search_seed={search_seed}")
    print(f"[hpo] GPUs=0-{NUM_GPUS-1}, modalities={args.modalities}, models={args.models}")

    all_trial_rows = []
    best_configs = {}

    for modality, model in configs_to_run:
        model_key = "effnet" if model in ("effnet", "effnet_b0") else "cnn"
        config_key = f"{modality}_{model_key}"
        print(f"\n[hpo] *** {modality}/{model} — {n_trials} trials ***")

        combo_seed = search_seed + abs(hash((modality, model))) % 1000
        rng = np.random.default_rng(combo_seed)

        all_hparams = [_sample_config(model_key, rng) for _ in range(n_trials)]

        trial_rows = []

        def _run_trial_wrapper(trial_idx):
            """Thread-pool worker — returns (trial_idx, row_or_None,"""
            hp = all_hparams[trial_idx]
            gpu_id = trial_idx % NUM_GPUS
            print(f"  trial {trial_idx+1}/{n_trials}: "
                  f"lr={hp['lr']:.2e} bs={hp['batch_size']} "
                  f"wd={hp['weight_decay']:.2e} aug={hp['aug_strength']} "
                  f"ls={hp['label_smoothing']} sampler={hp['sampler_mode']} "
                  f"patience={hp['patience']}"
                  + (f" widths={hp['cnn_widths']} p={hp['p']}" if model_key == "cnn" else "")
                  + f" -> GPU {gpu_id}", flush=True)
            row, elapsed = _run_one_trial(
                modality, model, trial_idx, hp,
                args.wall_cap_min, gpu_id, python_exe, search_seed=search_seed
            )
            return trial_idx, row, elapsed, gpu_id, hp

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
            futures = {executor.submit(_run_trial_wrapper, i): i for i in range(n_trials)}
            for fut in concurrent.futures.as_completed(futures):
                trial_idx, row, elapsed, gpu_id, hp = fut.result()
                if row is not None:
                    trial_row = {
                        "modality": modality,
                        "model": model_key,
                        "trial_idx": trial_idx,
                        "learning_rate": hp["lr"],
                        "batch_size": hp["batch_size"],
                        "weight_decay": hp["weight_decay"],
                        "aug_strength": hp["aug_strength"],
                        "label_smoothing": hp["label_smoothing"],
                        "imbalance": hp["sampler_mode"],
                        "patience": hp["patience"],
                        "cnn_widths": str(hp.get("cnn_widths")),
                        "p": hp.get("p", 0.3),
                        "best_val_score": row.get("best_val_score"),
                        "primary_metric_test": row.get("primary_metric"),
                        "epochs_ran": row.get("epochs_ran"),
                        "train_time_s": row.get("train_time_s"),
                        "gpu_id": gpu_id,
                        "elapsed_s": round(elapsed, 1),
                    }
                    trial_rows.append(trial_row)
                    all_trial_rows.append(trial_row)
                    print(f"  [trial {trial_idx+1} done] best_val_score={row.get('best_val_score'):.4f} "
                          f"primary_metric_test={row.get('primary_metric'):.4f} "
                          f"epochs={row.get('epochs_ran')} elapsed={elapsed:.0f}s", flush=True)
                else:
                    print(f"  [trial {trial_idx+1} FAILED] elapsed={elapsed:.0f}s", flush=True)

        trial_df = pd.DataFrame(all_trial_rows)
        trial_df.to_csv(HPO_CSV, index=False)

        if trial_rows:
            best_row = max(trial_rows, key=lambda r: r["best_val_score"])
            best_hparam = {
                "learning_rate": best_row["learning_rate"],
                "batch_size": best_row["batch_size"],
                "weight_decay": best_row["weight_decay"],
                "aug_strength": best_row["aug_strength"],
                "label_smoothing": best_row["label_smoothing"],
                "sampler_mode": best_row["imbalance"],
                "patience": best_row["patience"],
                "cnn_widths": eval(best_row["cnn_widths"]) if best_row["cnn_widths"] not in (None, "None") else None,
                "p": best_row["p"],
                "best_val_score": best_row["best_val_score"],
                "trial_idx": best_row["trial_idx"],
            }
            best_configs[config_key] = best_hparam
            print(f"\n[hpo] WINNER {config_key}: trial={best_row['trial_idx']} "
                  f"best_val_score={best_row['best_val_score']:.4f} "
                  f"(test={best_row['primary_metric_test']:.4f} — recorded but NOT used for selection)")
        else:
            print(f"\n[hpo] {config_key}: no successful trials — no winner.")

    if all_trial_rows:
        pd.DataFrame(all_trial_rows).to_csv(HPO_CSV, index=False)
        print(f"\n[wrote] {HPO_CSV} ({len(all_trial_rows)} trial rows)")

    with open(HPO_JSON, "w") as f:
        json.dump(best_configs, f, indent=2)
    print(f"[wrote] {HPO_JSON} ({len(best_configs)} best configs)")

    print("\n=== HPO LEADERBOARD (val-only selection) ===")
    if all_trial_rows:
        df = pd.DataFrame(all_trial_rows)
        for (modality, model), grp in df.groupby(["modality", "model"]):
            top = grp.sort_values("best_val_score", ascending=False).head(3)
            print(f"\n  {modality}/{model}:")
            for _, r in top.iterrows():
                print(f"    trial={int(r['trial_idx'])} "
                      f"best_val_score={r['best_val_score']:.4f} "
                      f"lr={r['learning_rate']:.2e} "
                      f"bs={int(r['batch_size'])} "
                      f"wd={r['weight_decay']:.2e}")
    print("=============================================")
    print("Done.")


if __name__ == "__main__":
    main()
