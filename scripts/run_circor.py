"""Run the CirCor murmur-detection sub-study (classical + deep, multi-seed).

CirCor caches share the heart schema, so the binary murmur task runs through the
heart code path: run_experiments("heart", ...) for the classical family and
run_modality(..., "heart") for the deep models, both aggregating windows to the
recording level for MAcc. Deep models are run across seeds (mean+/-std), one
subprocess per (model, seed) pinned to a GPU. Results go to dedicated circor
CSVs and are not merged into the heart/lung unified comparison.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

FEATURES_DIR = PROJECT_ROOT / "features"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
SPEC_CACHE = FEATURES_DIR / "circor_spectrograms.npy"
CLF_CACHE = FEATURES_DIR / "circor_classical.npy"

CLASSICAL_CSV = TABLES_DIR / "metrics_circor_classical.csv"
DEEP_CSV = TABLES_DIR / "circor_deep_multiseed.csv"
DEEP_RAW_CSV = TABLES_DIR / "circor_deep_raw.csv"
COMPARISON_CSV = TABLES_DIR / "circor_comparison.csv"


def _load(path):
    return np.load(path, allow_pickle=True).item()


def _run_worker(seed, arch, wall_cap_s, out_json):
    from src.train_cnn import run_modality
    cache = _load(SPEC_CACHE)
    out_dir = FIGURES_DIR / f"circor_{arch}_seed{seed}"
    row = run_modality(cache, "heart", model=arch, wall_cap_s=wall_cap_s,
                       seed=seed, out_dir=str(out_dir))
    keep = {k: v for k, v in row.items()
            if isinstance(v, (int, float, str, bool, type(None)))}
    keep["seed"] = seed
    keep["arch"] = "effnet_b0" if arch in ("effnet", "effnet_b0") else "cnn"
    Path(out_json).write_text(json.dumps(keep))
    print(f"[circor worker seed={seed} arch={arch}] "
          f"{keep.get('primary_metric_name')}={keep.get('primary_metric')}")


def _run_classical():
    from src.train_classical import run_experiments
    clf = _load(CLF_CACHE)
    fig_dir = FIGURES_DIR / "circor_classical"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = run_experiments("heart", clf, figures_dir=str(fig_dir))
    df = pd.DataFrame(rows)
    df["modality"] = "circor"
    df = df[["modality"] + [c for c in df.columns if c != "modality"]]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLASSICAL_CSV, index=False)
    print(f"[wrote] {CLASSICAL_CSV} ({len(df)} rows)")
    return df


def _aggregate_deep(raw_df):
    out = []
    for (arch,), grp in raw_df.groupby(["arch"], sort=False):
        n = len(grp)
        rec = {"model": arch,
               "primary_metric_name": grp["primary_metric_name"].iloc[0]}
        for col in ["primary_metric", "Se", "Sp", "macro_f1", "auc_roc", "accuracy"]:
            if col in grp.columns:
                vals = pd.to_numeric(grp[col], errors="coerce")
                rec[f"{col}_mean"] = round(float(vals.mean()), 4)
                rec[f"{col}_std"] = round(float(vals.std(ddof=1)), 4) if n > 1 else 0.0
        rec["n_seeds"] = n
        rec["seeds"] = str(sorted(int(s) for s in grp["seed"].tolist()))
        rec["n_train"] = grp["n_train"].iloc[0] if "n_train" in grp.columns else ""
        rec["n_test"] = grp["n_test"].iloc[0] if "n_test" in grp.columns else ""
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="CirCor murmur sub-study runner.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--archs", nargs="+", default=["cnn", "effnet"],
                    choices=["cnn", "effnet"])
    ap.add_argument("--wall-cap-min", type=float, default=30.0)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--arch", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-json", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    wall_cap_s = int(args.wall_cap_min * 60)

    if args.worker:
        _run_worker(args.seed, args.arch, wall_cap_s, args.out_json)
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PROJECT_ROOT / "results" / "_circor_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    classical_df = None
    if not args.skip_classical:
        print("[circor] running classical experiments (CPU) ...")
        classical_df = _run_classical()

    print(f"[circor] launching deep jobs: archs={args.archs} seeds={args.seeds} "
          f"gpus={args.gpus}")
    jobs = [(s, a) for s in args.seeds for a in args.archs]
    procs = []
    for i, (seed, arch) in enumerate(jobs):
        gpu = args.gpus[i % len(args.gpus)]
        out_json = tmp / f"circor_s{seed}_{arch}.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--seed", str(seed), "--arch", arch,
               "--wall-cap-min", str(args.wall_cap_min), "--out-json", str(out_json)]
        log = open(tmp / f"circor_s{seed}_{arch}.log", "w")
        print(f"[circor] launch seed={seed} arch={arch} -> GPU {gpu}")
        procs.append((seed, arch, out_json,
                      subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    t0 = time.time()
    deep_rows = []
    for seed, arch, out_json, p, log in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            print(f"[circor] seed={seed} arch={arch} FAILED rc={rc}; see {tmp}/circor_s{seed}_{arch}.log")
            continue
        deep_rows.append(json.loads(Path(out_json).read_text()))
        print(f"[circor] seed={seed} arch={arch} done")

    if deep_rows:
        raw = pd.DataFrame(deep_rows)
        raw.to_csv(DEEP_RAW_CSV, index=False)
        deep_summary = _aggregate_deep(raw)
        deep_summary.to_csv(DEEP_CSV, index=False)
        print(f"[wrote] {DEEP_CSV} ({len(deep_summary)} rows, elapsed {time.time()-t0:.0f}s)")
    else:
        deep_summary = pd.DataFrame()
        print("[circor] no deep rows collected.")

    # Unified circor comparison: classical (best per feature_set/model) + deep means.
    comp = []
    if classical_df is not None:
        for r in classical_df.itertuples(index=False):
            comp.append({
                "family": "classical",
                "model": f"{r.model}-{r.feature_set}" if hasattr(r, "feature_set") else r.model,
                "primary_metric_name": "MAcc",
                "primary_metric": getattr(r, "primary_metric", ""),
                "std": "",
                "Se": getattr(r, "Se", ""), "Sp": getattr(r, "Sp", ""),
                "auc_roc": getattr(r, "auc_roc", ""),
            })
    for r in deep_summary.itertuples(index=False):
        comp.append({
            "family": "deep",
            "model": r.model,
            "primary_metric_name": getattr(r, "primary_metric_name", "MAcc"),
            "primary_metric": getattr(r, "primary_metric_mean", ""),
            "std": getattr(r, "primary_metric_std", ""),
            "Se": getattr(r, "Se_mean", ""), "Sp": getattr(r, "Sp_mean", ""),
            "auc_roc": getattr(r, "auc_roc_mean", ""),
        })
    pd.DataFrame(comp).to_csv(COMPARISON_CSV, index=False)
    print(f"[wrote] {COMPARISON_CSV} ({len(comp)} rows)")

    print("\n=== CIRCOR SUMMARY ===")
    for c in comp:
        s = f"+/-{c['std']}" if c["std"] != "" else ""
        print(f"  {c['family']:9s} {c['model']:22s} {c['primary_metric_name']}={c['primary_metric']}{s}")
    print("======================")


if __name__ == "__main__":
    main()
