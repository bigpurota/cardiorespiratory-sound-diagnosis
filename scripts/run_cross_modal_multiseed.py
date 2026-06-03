"""Multi-seed cross-modal battery for mean+/-std error bars.

Runs the full cross-modal experiment set (in-domain heart/lung, transfer in
both directions, and joint multi-task) across several seeds and aggregates the
primary metric per (setting, source, target, model) into mean+/-std. Mirrors
the seeded methodology of the in-domain multi-seed run so Chapter 4 numbers are
consistent with Chapter 3.

Orchestrator mode spawns one worker subprocess per seed, each pinned to its own
GPU via CUDA_VISIBLE_DEVICES; worker mode runs one seed and writes its rows to a
JSON file.
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
SUMMARY_CSV = TABLES_DIR / "cross_modal_multiseed.csv"
RAW_CSV = TABLES_DIR / "cross_modal_multiseed_raw.csv"

KEY_COLUMNS = ["setting", "source_modality", "target_modality", "model"]
METRIC_COLUMNS = ["primary_metric", "Se", "Sp", "macro_f1", "accuracy"]


def _load_cache(modality):
    path = FEATURES_DIR / f"{modality}_spectrograms.npy"
    if not path.exists():
        raise FileNotFoundError(f"spectrogram cache missing: {path}")
    return np.load(path, allow_pickle=True).item()


def run_battery(seed, archs, wall_cap_s):
    """Run the cross-modal battery for one seed; return a list of row dicts."""
    from src.split import assert_no_patient_leakage
    from src.train_cnn import run_modality
    from src.cross_modal import transfer_modality, train_joint, evaluate_joint
    from src.datasets import build_loaders

    heart_cache = _load_cache("heart")
    lung_cache = _load_cache("lung")

    for cache, mod in ((heart_cache, "heart"), (lung_cache, "lung")):
        pid = np.asarray(list(map(str, cache["patient_id"])), dtype=object)
        spl = np.asarray(cache["split"], dtype=object)
        assert_no_patient_leakage(pid[spl == "train"], pid[spl == "test"])

    rows = []
    for arch in archs:
        for_effnet = arch in ("effnet", "effnet_b0", "efficientnet")
        model_label = "effnet_b0" if for_effnet else "cnn"
        out_base = FIGURES_DIR / f"crossmodal_ms_{model_label}_seed{seed}"
        out_base.mkdir(parents=True, exist_ok=True)

        def _row(d, setting, src, tgt):
            return {
                "seed": seed,
                "setting": setting,
                "source_modality": src,
                "target_modality": tgt,
                "model": d.get("model", model_label),
                "primary_metric_name": d.get("primary_metric_name", ""),
                "primary_metric": float(d.get("primary_metric", float("nan"))),
                "Se": float(d.get("Se", float("nan"))),
                "Sp": float(d.get("Sp", float("nan"))),
                "macro_f1": float(d.get("macro_f1", float("nan"))),
                "accuracy": float(d.get("accuracy", float("nan"))),
                "n_train": d.get("n_train", 0),
                "n_test": d.get("n_test", 0),
            }

        # In-domain references.
        rh = run_modality(heart_cache, "heart", model=arch, wall_cap_s=wall_cap_s,
                          seed=seed, out_dir=str(out_base / "in_domain_heart"))
        rows.append(_row(rh, "in_domain", "heart", "heart"))

        rl = run_modality(lung_cache, "lung", model=arch, wall_cap_s=wall_cap_s,
                          seed=seed, out_dir=str(out_base / "in_domain_lung"))
        rows.append(_row(rl, "in_domain", "lung", "lung"))

        # Cross-modal transfer, both directions.
        h2l = transfer_modality(source_cache=heart_cache, target_cache=lung_cache,
                                source_modality="heart", target_modality="lung",
                                arch=arch, wall_cap_s=wall_cap_s, seed=seed,
                                out_dir=str(out_base / "transfer_heart_lung"))
        rows.append(_row(h2l, "transfer", "heart", "lung"))

        l2h = transfer_modality(source_cache=lung_cache, target_cache=heart_cache,
                                source_modality="lung", target_modality="heart",
                                arch=arch, wall_cap_s=wall_cap_s, seed=seed,
                                out_dir=str(out_base / "transfer_lung_heart"))
        rows.append(_row(l2h, "transfer", "lung", "heart"))

        # Joint multi-task.
        joint_model, _info = train_joint(heart_cache=heart_cache, lung_cache=lung_cache,
                                         arch=arch, wall_cap_s=wall_cap_s, seed=seed,
                                         out_dir=str(out_base / "joint"))
        heart_loaders = build_loaders(heart_cache, "heart", for_effnet=for_effnet,
                                      batch_size=32, seed=seed)
        lung_loaders = build_loaders(lung_cache, "lung", for_effnet=for_effnet,
                                     batch_size=32, seed=seed)
        joint_rows = evaluate_joint(joint_model, heart_loaders=heart_loaders,
                                    lung_loaders=lung_loaders,
                                    out_dir=str(out_base / "joint"),
                                    model_name=model_label)
        for jr in joint_rows:
            rows.append(_row(jr, "joint", "heart+lung", jr["target_modality"]))

    return rows


def _aggregate(raw_df):
    """Aggregate per-seed rows into mean+/-std per experiment key."""
    out = []
    for key, grp in raw_df.groupby(KEY_COLUMNS, sort=False):
        n = len(grp)
        rec = dict(zip(KEY_COLUMNS, key))
        rec["primary_metric_name"] = grp["primary_metric_name"].iloc[0]
        for col in METRIC_COLUMNS:
            vals = pd.to_numeric(grp[col], errors="coerce")
            rec[f"{col}_mean"] = round(float(vals.mean()), 4)
            rec[f"{col}_std"] = round(float(vals.std(ddof=1)), 4) if n > 1 else 0.0
        rec["n_seeds"] = n
        rec["seeds"] = str(sorted(int(s) for s in grp["seed"].tolist()))
        rec["n_train"] = grp["n_train"].iloc[0]
        rec["n_test"] = grp["n_test"].iloc[0]
        out.append(rec)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="Multi-seed cross-modal battery (mean+/-std).")
    ap.add_argument("--arch", default="both", choices=["cnn", "effnet", "both"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--wall-cap-min", type=float, default=30.0)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2],
                    help="GPU ids to spread seed-workers across.")
    # Worker mode.
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-json", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    archs = ["cnn", "effnet"] if args.arch == "both" else [args.arch]
    wall_cap_s = int(args.wall_cap_min * 60)

    if args.worker:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[worker seed={args.seed}] device={dev} "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} archs={archs}")
        rows = run_battery(args.seed, archs, wall_cap_s)
        Path(args.out_json).write_text(json.dumps(rows))
        print(f"[worker seed={args.seed}] wrote {len(rows)} rows -> {args.out_json}")
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = PROJECT_ROOT / "results" / "_cm_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[orchestrator] seeds={args.seeds} archs={archs} "
          f"wall_cap={args.wall_cap_min}min gpus={args.gpus}")

    # Split into one job per (seed, arch) so cnn and effnet for a seed run on
    # separate GPUs concurrently. The battery is CPU-bound (data loading), so
    # packing 6 jobs onto 4 GPUs on a 60-core box maximises throughput.
    jobs = [(seed, a) for seed in args.seeds for a in archs]
    procs = []
    for i, (seed, a) in enumerate(jobs):
        gpu = args.gpus[i % len(args.gpus)]
        out_json = tmp_dir / f"cm_seed{seed}_{a}.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "8"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--worker", "--seed", str(seed), "--arch", a,
               "--wall-cap-min", str(args.wall_cap_min),
               "--out-json", str(out_json)]
        print(f"[orchestrator] launch seed={seed} arch={a} -> GPU {gpu}")
        log = open(tmp_dir / f"cm_seed{seed}_{a}.log", "w")
        procs.append((seed, a, out_json,
                      subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    t0 = time.time()
    all_rows = []
    for seed, a, out_json, p, log in procs:
        ret = p.wait()
        log.close()
        if ret != 0:
            print(f"[orchestrator] seed={seed} arch={a} FAILED (rc={ret}); "
                  f"see {tmp_dir}/cm_seed{seed}_{a}.log")
            continue
        rows = json.loads(Path(out_json).read_text())
        all_rows.extend(rows)
        print(f"[orchestrator] seed={seed} arch={a} collected {len(rows)} rows")

    if not all_rows:
        print("[orchestrator] no rows collected; aborting.")
        sys.exit(1)

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(RAW_CSV, index=False)
    print(f"[wrote] {RAW_CSV} ({len(raw_df)} rows, elapsed {time.time()-t0:.0f}s)")

    summary = _aggregate(raw_df)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[wrote] {SUMMARY_CSV} ({len(summary)} rows)")

    print("\n=== CROSS-MODAL MULTI-SEED SUMMARY ===")
    for _, r in summary.iterrows():
        print(f"  {r['setting']:9s} {r['source_modality']:10s}->{r['target_modality']:5s} "
              f"{r['model']:10s} {r['primary_metric_name']:11s}="
              f"{r['primary_metric_mean']:.4f}+/-{r['primary_metric_std']:.4f} "
              f"(n={r['n_seeds']})")
    print("======================================")


if __name__ == "__main__":
    main()
