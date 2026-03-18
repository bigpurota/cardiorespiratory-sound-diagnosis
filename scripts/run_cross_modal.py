#!/usr/bin/env python
"""
scripts/run_cross_modal.py — GPU CLI for cross-modal transfer + joint multi-task.

Runs all 5 experiment cells and writes the cross-modal summary artefacts:
  - IN-DOMAIN baselines (2):   heart→heart, lung→lung (via run_modality)
  - CROSS-DOMAIN transfer (2): heart→lung, lung→heart (via transfer_modality)
  - JOINT multi-task (1 model → 2 rows): train_joint → evaluate_joint

Outputs (DO NOT modify unified_comparison.csv — it is READ-ONLY here):
  results/tables/cross_modal_summary.csv   — one row per cell, idempotent
  results/tables/cross_modal_spearman.csv  — Spearman rho from unified_comparison.csv
  results/figures/cross_modal_heatmap.png  — source×target transfer-matrix heatmap

Device auto-detect (train_one_model / train_joint both call torch.cuda.is_available())
so the SAME script runs on CPU (tiny smoke) and the 4×A100 box (full run).

Usage:
    uv run python scripts/run_cross_modal.py --arch cnn --wall-cap-min 1 --seed 42
    # Full GPU run (from the rented box):
    OMP_NUM_THREADS=8 .venv/bin/python scripts/run_cross_modal.py --arch both \\
        --wall-cap-min 45 --seed 42
"""
import os

# macOS duplicate-OpenMP-runtime guard (VERBATIM from src/train_cnn.py / scripts/run_cnn.py):
# MUST run BEFORE the first ``import torch``.  ``setdefault`` lets a caller override.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402,F401 — import FIRST (seeds RNGs, exposes paths)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # MUST precede pyplot import — headless (Pitfall 7)
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402

from src.split import assert_no_patient_leakage  # noqa: E402
from src.train_cnn import run_modality as _run_modality_single  # noqa: E402
from src.cross_modal import (  # noqa: E402
    transfer_modality,
    train_joint,
    evaluate_joint,
    spearman_method_rankings,
    build_shared_encoder,
)
from src.datasets import build_loaders  # noqa: E402

FEATURES_DIR = os.path.join(config.PROJECT_ROOT, "features")
TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(config.RESULTS_DIR, "figures")

SUMMARY_CSV = os.path.join(TABLES_DIR, "cross_modal_summary.csv")
SPEARMAN_CSV = os.path.join(TABLES_DIR, "cross_modal_spearman.csv")
HEATMAP_PNG = os.path.join(FIGURES_DIR, "cross_modal_heatmap.png")
UNIFIED_CSV = os.path.join(TABLES_DIR, "unified_comparison.csv")

SUMMARY_COLUMNS = [
    "setting", "source_modality", "target_modality", "model",
    "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "accuracy",
    "n_train", "n_test",
]


# ---------------------------------------------------------------------------
# Cache loader (mirrors scripts/run_cnn.py::_load_cache)
# ---------------------------------------------------------------------------

def _load_cache(modality):
    """np.load the spectrogram cache; build it if absent."""
    path = os.path.join(FEATURES_DIR, f"{modality}_spectrograms.npy")
    if not os.path.exists(path):
        print(
            f"[run_cross_modal] cache missing: {path} — building via "
            f"scripts/build_spectrograms.py --modality {modality} ..."
        )
        from scripts.build_spectrograms import build as build_spectrograms
        build_spectrograms(modality)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"spectrogram cache still missing after build: {path}"
        )
    return np.load(path, allow_pickle=True).item()


# ---------------------------------------------------------------------------
# Idempotent summary CSV writer (drop-mask by (setting, source, target, model))
# ---------------------------------------------------------------------------

def _write_summary(rows):
    """Write/merge cross_modal_summary.csv idempotently.

    Drops rows matching the (setting, source_modality, target_modality, model) keys
    produced by this run, then appends the new rows — re-runs cannot duplicate rows.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    new_df = pd.DataFrame(
        [{c: r.get(c, "") for c in SUMMARY_COLUMNS} for r in rows]
    )

    if os.path.exists(SUMMARY_CSV):
        existing = pd.read_csv(SUMMARY_CSV)
        for c in SUMMARY_COLUMNS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[SUMMARY_COLUMNS]
        # Drop only the keys produced by this run.
        keys = new_df[["setting", "source_modality", "target_modality", "model"]].apply(
            tuple, axis=1
        )
        existing_keys = existing[
            ["setting", "source_modality", "target_modality", "model"]
        ].apply(tuple, axis=1)
        existing = existing[~existing_keys.isin(set(keys))]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined[SUMMARY_COLUMNS]
    combined.to_csv(SUMMARY_CSV, index=False)
    print(f"[wrote] {SUMMARY_CSV} ({len(combined)} rows)")


# ---------------------------------------------------------------------------
# Spearman CSV writer
# ---------------------------------------------------------------------------

def _write_spearman(rho, pvalue, n, labels, heart_scores, lung_scores):
    os.makedirs(TABLES_DIR, exist_ok=True)
    df = pd.DataFrame([{
        "rho": rho,
        "pvalue": pvalue,
        "n_methods": n,
        "methods": "|".join(str(m) for m in labels),
        "heart_scores": "|".join(f"{s:.6f}" for s in heart_scores),
        "lung_scores": "|".join(f"{s:.6f}" for s in lung_scores),
        "feature_set": "A_mfcc_delta",
    }])
    df.to_csv(SPEARMAN_CSV, index=False)
    print(f"[wrote] {SPEARMAN_CSV}  rho={rho:.4f}  pvalue={pvalue:.4f}")


# ---------------------------------------------------------------------------
# Heatmap renderer
# ---------------------------------------------------------------------------

def _render_heatmap(rows):
    """Render a source×target primary-metric heatmap and save to HEATMAP_PNG."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Build matrix: rows = source modality (heart / lung / heart+lung),
    #               cols = target modality (heart / lung).
    sources = ["heart", "lung", "heart+lung"]
    targets = ["heart", "lung"]

    # Index rows by (source, target) → primary_metric.
    lookup = {}
    for r in rows:
        key = (r.get("source_modality", ""), r.get("target_modality", ""))
        # For joint rows, take either heart or lung target.
        existing = lookup.get(key)
        if existing is None or r.get("setting") != "in_domain":
            lookup[key] = r.get("primary_metric", float("nan"))

    matrix = []
    metric_names = []
    for src in sources:
        row_vals = []
        for tgt in targets:
            val = lookup.get((src, tgt), float("nan"))
            row_vals.append(val)
            # Track metric name for the label.
            for r in rows:
                if r.get("source_modality") == src and r.get("target_modality") == tgt:
                    metric_names.append(r.get("primary_metric_name", ""))
        matrix.append(row_vals)

    matrix_arr = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix_arr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=11)
    ax.set_xlabel("Target (evaluation) modality", fontsize=11)
    ax.set_ylabel("Source (training) modality", fontsize=11)
    ax.set_title("Cross-Modal Transfer Matrix — Primary Metric", fontsize=12)

    # Annotate cells with the numeric value.
    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            val = matrix_arr[i, j]
            txt = f"{val:.3f}" if not np.isnan(val) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="black" if 0.3 < val < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Primary metric (MAcc / ICBHI Score)")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=150)
    plt.close(fig)
    print(f"[wrote] {HEATMAP_PNG}")


# ---------------------------------------------------------------------------
# In-domain baseline helper (wraps run_modality from src.train_cnn)
# ---------------------------------------------------------------------------

def _run_in_domain(cache, modality, arch, wall_cap_s, seed, out_dir):
    """Run a single in-domain experiment via src.train_cnn.run_modality."""
    row = _run_modality_single(
        cache, modality, model=arch,
        wall_cap_s=wall_cap_s, seed=seed, out_dir=out_dir,
    )
    # Shape the row to match cross_modal_summary schema.
    return {
        "setting": "in_domain",
        "source_modality": modality,
        "target_modality": modality,
        "model": row.get("model", arch),
        "primary_metric_name": row.get("primary_metric_name", ""),
        "primary_metric": row.get("primary_metric", float("nan")),
        "Se": row.get("Se", float("nan")),
        "Sp": row.get("Sp", float("nan")),
        "macro_f1": row.get("macro_f1", float("nan")),
        "accuracy": row.get("accuracy", float("nan")),
        "n_train": row.get("n_train", 0),
        "n_test": row.get("n_test", 0),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run cross-modal transfer + joint multi-task experiments "
            "and write summary CSV + heatmap + Spearman CSV."
        )
    )
    ap.add_argument(
        "--arch", default="cnn", choices=["cnn", "effnet", "both"],
        help="Model architecture: cnn | effnet | both (default cnn for CPU smoke).",
    )
    ap.add_argument(
        "--wall-cap-min", type=float, default=20.0,
        help="Per-experiment wall-clock cap in minutes (D-03; relax on GPU).",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42).")
    ap.add_argument(
        "--out-dir", default=None,
        help="Override output directory for figures/checkpoints (optional).",
    )
    args = ap.parse_args()

    wall_cap_s = int(args.wall_cap_min * 60)
    seed = args.seed
    archs = ["cnn", "effnet"] if args.arch == "both" else [args.arch]

    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_cross_modal] device={device_str}  archs={archs}  wall_cap_s={wall_cap_s}  seed={seed}")

    # --- Load + leakage-assert BOTH caches at startup ---
    print("[run_cross_modal] loading heart cache ...")
    heart_cache = _load_cache("heart")
    print("[run_cross_modal] loading lung cache ...")
    lung_cache = _load_cache("lung")

    # Startup leakage re-assert on both (logs [leakage-check OK] × 2).
    for cache, mod in ((heart_cache, "heart"), (lung_cache, "lung")):
        pid = np.asarray(list(map(str, cache["patient_id"])), dtype=object)
        spl = np.asarray(cache["split"], dtype=object)
        assert_no_patient_leakage(pid[spl == "train"], pid[spl == "test"])

    all_rows = []

    for arch in archs:
        for_effnet = arch in ("effnet", "effnet_b0", "efficientnet")
        model_label = "effnet_b0" if for_effnet else "cnn"
        out_base = args.out_dir or os.path.join(FIGURES_DIR, f"crossmodal_{model_label}")
        os.makedirs(out_base, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[arch={arch}]  wall_cap_s={wall_cap_s}")
        print(f"{'='*60}")

        # --- IN-DOMAIN baselines ---
        print(f"\n[1/5] IN-DOMAIN heart ({arch}) ...")
        try:
            row_h = _run_in_domain(
                heart_cache, "heart", arch, wall_cap_s, seed,
                os.path.join(out_base, "in_domain_heart"),
            )
            print(f"      heart in-domain  {row_h['primary_metric_name']}={row_h['primary_metric']:.4f}")
            all_rows.append(row_h)
        except Exception as e:
            print(f"      [WARN] heart in-domain failed: {e}")

        print(f"\n[2/5] IN-DOMAIN lung ({arch}) ...")
        try:
            row_l = _run_in_domain(
                lung_cache, "lung", arch, wall_cap_s, seed,
                os.path.join(out_base, "in_domain_lung"),
            )
            print(f"      lung in-domain  {row_l['primary_metric_name']}={row_l['primary_metric']:.4f}")
            all_rows.append(row_l)
        except Exception as e:
            print(f"      [WARN] lung in-domain failed: {e}")

        # --- CROSS-DOMAIN: heart → lung ---
        print(f"\n[3/5] TRANSFER heart→lung ({arch}) ...")
        try:
            row_h2l = transfer_modality(
                source_cache=heart_cache,
                target_cache=lung_cache,
                source_modality="heart",
                target_modality="lung",
                arch=arch,
                wall_cap_s=wall_cap_s,
                seed=seed,
                out_dir=os.path.join(out_base, "transfer_heart_lung"),
            )
            print(f"      heart→lung  {row_h2l['primary_metric_name']}={row_h2l['primary_metric']:.4f}")
            # Honest reporting: flag weak/negative transfer.
            in_domain_lung_metric = next(
                (r["primary_metric"] for r in all_rows
                 if r["setting"] == "in_domain" and r["target_modality"] == "lung"
                 and r["model"] == model_label),
                None
            )
            if in_domain_lung_metric is not None and row_h2l["primary_metric"] < in_domain_lung_metric:
                print(f"      NOTE: heart→lung transfer WEAK/NEGATIVE "
                      f"({row_h2l['primary_metric']:.4f} < in-domain {in_domain_lung_metric:.4f}) "
                      f"— recording real number, not inflated (honest reporting).")
            all_rows.append(row_h2l)
        except Exception as e:
            print(f"      [WARN] heart→lung transfer failed: {e}")

        # --- CROSS-DOMAIN: lung → heart ---
        print(f"\n[4/5] TRANSFER lung→heart ({arch}) ...")
        try:
            row_l2h = transfer_modality(
                source_cache=lung_cache,
                target_cache=heart_cache,
                source_modality="lung",
                target_modality="heart",
                arch=arch,
                wall_cap_s=wall_cap_s,
                seed=seed,
                out_dir=os.path.join(out_base, "transfer_lung_heart"),
            )
            print(f"      lung→heart  {row_l2h['primary_metric_name']}={row_l2h['primary_metric']:.4f}")
            in_domain_heart_metric = next(
                (r["primary_metric"] for r in all_rows
                 if r["setting"] == "in_domain" and r["target_modality"] == "heart"
                 and r["model"] == model_label),
                None
            )
            if in_domain_heart_metric is not None and row_l2h["primary_metric"] < in_domain_heart_metric:
                print(f"      NOTE: lung→heart transfer WEAK/NEGATIVE "
                      f"({row_l2h['primary_metric']:.4f} < in-domain {in_domain_heart_metric:.4f}) "
                      f"— recording real number, not inflated (honest reporting).")
            all_rows.append(row_l2h)
        except Exception as e:
            print(f"      [WARN] lung→heart transfer failed: {e}")

        # --- JOINT multi-task ---
        print(f"\n[5/5] JOINT multi-task ({arch}) ...")
        try:
            joint_model, train_info = train_joint(
                heart_cache=heart_cache,
                lung_cache=lung_cache,
                arch=arch,
                wall_cap_s=wall_cap_s,
                seed=seed,
                out_dir=os.path.join(out_base, "joint"),
            )
            print(f"      joint best_val_score={train_info['best_val_score']:.4f}  "
                  f"epochs={train_info['epochs_ran']}")

            # Build loaders for evaluation (same loaders as training — reuse build_loaders).
            heart_loaders = build_loaders(
                heart_cache, "heart",
                for_effnet=for_effnet, batch_size=32, seed=seed,
            )
            lung_loaders = build_loaders(
                lung_cache, "lung",
                for_effnet=for_effnet, batch_size=32, seed=seed,
            )

            joint_rows = evaluate_joint(
                joint_model,
                heart_loaders=heart_loaders,
                lung_loaders=lung_loaders,
                out_dir=os.path.join(out_base, "joint"),
                model_name=model_label,
            )
            for jr in joint_rows:
                print(f"      joint {jr['target_modality']}  "
                      f"{jr['primary_metric_name']}={jr['primary_metric']:.4f}")
            all_rows.extend(joint_rows)
        except Exception as e:
            print(f"      [WARN] joint multi-task failed: {e}")

    # --- Write cross_modal_summary.csv ---
    _write_summary(all_rows)

    # --- Spearman rank correlation (reads unified_comparison.csv — READ-ONLY there) ---
    if os.path.exists(UNIFIED_CSV):
        try:
            rho, pvalue, n, labels, h_scores, l_scores = spearman_method_rankings(UNIFIED_CSV)
            print(f"\n[Spearman] rho={rho:.4f}  pvalue={pvalue:.4f}  n={n}  methods={labels}")
            _write_spearman(rho, pvalue, n, labels, h_scores, l_scores)
        except Exception as e:
            print(f"[WARN] Spearman computation failed: {e}")
    else:
        print(f"[WARN] {UNIFIED_CSV} not found — skipping Spearman (unified_comparison.csv required).")

    # --- Heatmap ---
    if all_rows:
        _render_heatmap(all_rows)

    # --- Verify unified_comparison.csv is untouched ---
    if os.path.exists(UNIFIED_CSV):
        n_unified = len(pd.read_csv(UNIFIED_CSV))
        print(f"\n[verify] unified_comparison.csv still has {n_unified} rows (untouched).")
        if n_unified != 20:
            print(f"  [WARN] expected 20 rows; got {n_unified} — check for unintended modifications.")

    print("\n[run_cross_modal] DONE.")
    sys.exit(0)


if __name__ == "__main__":
    main()
