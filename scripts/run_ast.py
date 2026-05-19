#!/usr/bin/env python
"""
Fine-tune a pretrained Audio Spectrogram Transformer (AST) on heart and lung sounds.

Fine-tunes ``MIT/ast-finetuned-audioset-10-10-0.4593`` on both modalities using the
log-mel spectrogram caches (``features/{modality}_spectrograms.npy``), reusing the same
leakage-safe loaders and metrics as the CNN/EfficientNet experiments so the AST rows are
directly comparable in ``unified_comparison.csv``. Heart is headlined on MAcc, lung on
the ICBHI Score.

Any per-modality failure (download failure, OOM, non-convergence) is caught and written
as a limitation row in ``metrics_ast.csv`` with a NaN primary metric; it is not merged
into ``unified_comparison.csv``, so the unified table never carries a fabricated number.

    uv run python scripts/run_ast.py --modality heart
    uv run python scripts/run_ast.py --modality all --wall-cap-min 45
"""
import os

# macOS duplicate-OpenMP-runtime guard: torch and sklearn/xgboost each bundle their own
# libomp.dylib; capping OpenMP threads and allowing the duplicate runtime avoids the
# collision segfault. Must run before the first ``import torch``.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402,F401 — import FIRST (seeds RNGs, exposes paths)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.split import assert_no_patient_leakage  # noqa: E402
from src.datasets import build_loaders  # noqa: E402
from src.ast_model import build_ast, count_params  # noqa: E402
from src.train_cnn import train_one_model, evaluate, _val_macc, _val_icbhi  # noqa: E402

FEATURES_DIR = os.path.join(config.PROJECT_ROOT, "features")
TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(config.RESULTS_DIR, "figures")

UNIFIED_CSV = os.path.join(TABLES_DIR, "unified_comparison.csv")
VOLUMETRICS_CSV = os.path.join(TABLES_DIR, "volumetrics_cnn.csv")

# unified_comparison.csv column order (12-column schema shared with the other drivers).
UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]

# volumetrics_cnn.csv column order.
VOLUMETRICS_COLUMNS = [
    "modality", "feature_set", "model", "train_time_s",
    "n_train_segments", "n_test_segments",
    "n_train_recordings", "n_test_recordings",
    "n_train_patients", "n_test_patients",
    "params", "fallback_from", "data_volume_mb",
]

# Per-modality metrics CSV columns (full suite + a fallback_from provenance column).
METRICS_COLUMNS = {
    "heart": [
        "feature_set", "model", "primary_metric_name", "primary_metric",
        "MAcc", "Se", "Sp", "macro_f1", "auc_roc", "accuracy",
        "n_train", "n_test", "params", "fallback_from", "cm_figure",
    ],
    "lung": [
        "feature_set", "model", "primary_metric_name", "primary_metric",
        "ICBHI_Score", "Se", "Sp", "macro_f1", "accuracy",
        "se_crackle", "se_wheeze", "se_both", "se_normal",
        "n_train", "n_test", "params", "fallback_from", "cm_figure",
    ],
}

# Chance-level threshold below which a run is judged non-converged.
_CHANCE_THRESHOLD = {"heart": 0.5, "lung": 0.5}

# Feature-set label for AST rows in unified_comparison.csv.
_AST_FS_LABEL = "log_mel_64x128"

# Default fine-tune learning rate (transformers fine-tune much lower than a CNN).
_DEFAULT_LR = 1e-5


def _load_cache(modality):
    """Load the spectrogram cache for ``modality``, building it if absent."""
    path = os.path.join(FEATURES_DIR, f"{modality}_spectrograms.npy")
    if not os.path.exists(path):
        print(
            f"[run_ast] spectrogram cache missing: {path} — building it via "
            f"scripts/build_spectrograms.py --modality {modality} ..."
        )
        from scripts.build_spectrograms import build as build_spectrograms
        build_spectrograms(modality)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"spectrogram cache still missing after build: {path} — run "
            f"`uv run python scripts/build_spectrograms.py --modality {modality}` first."
        )
    return np.load(path, allow_pickle=True).item(), path


def _write_metrics_csv(modality, rows):
    """Write ``results/tables/metrics_ast.csv`` (creates or appends per modality)."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    cols = METRICS_COLUMNS[modality]
    df_new = pd.DataFrame(rows)
    for c in cols:
        if c not in df_new.columns:
            df_new[c] = ""
    df_new = df_new[cols]

    out = os.path.join(TABLES_DIR, "metrics_ast.csv")
    if os.path.exists(out):
        existing = pd.read_csv(out)
        # Drop existing rows for this modality so re-runs are idempotent.
        if "modality" in existing.columns:
            existing = existing[existing["modality"] != modality]
        # Add modality column to new rows for tracking.
        df_new.insert(0, "modality", modality)
        existing_has_modality = "modality" in pd.read_csv(out).columns
        if existing_has_modality:
            combined = pd.concat([existing, df_new], ignore_index=True)
        else:
            existing.insert(0, "modality", "")
            combined = pd.concat([existing, df_new], ignore_index=True)
        combined.to_csv(out, index=False)
    else:
        df_new.insert(0, "modality", modality)
        df_new.to_csv(out, index=False)
    print(f"[wrote] {out} ({len(df_new)} new rows for {modality})")
    return out


def _rebuild_unified(modality, rows):
    """Idempotently merge AST rows into unified_comparison.csv.

    The drop-mask targets only ``model=='ast'`` for the current modality, so the
    classical and CNN/EffNet rows survive untouched and re-runs are idempotent.
    """
    new = pd.DataFrame([{c: r.get(c, "") for c in UNIFIED_COLUMNS} for r in rows])

    if os.path.exists(UNIFIED_CSV):
        existing = pd.read_csv(UNIFIED_CSV)
        for c in UNIFIED_COLUMNS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[UNIFIED_COLUMNS]
        # Drop only this modality's AST rows so re-runs are idempotent.
        drop_mask = (
            (existing["modality"] == modality)
            & (existing["model"] == "ast")
        )
        existing = existing[~drop_mask]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined = combined[UNIFIED_COLUMNS]
    combined.to_csv(UNIFIED_CSV, index=False)
    n_ast = int((combined["model"] == "ast").sum())
    print(f"[wrote] {UNIFIED_CSV} ({len(combined)} rows; {n_ast} AST)")
    return UNIFIED_CSV


def _rebuild_volumetrics(modality, rows, cache_path):
    """Merge this modality's AST volumetrics into volumetrics_cnn.csv."""
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
        # Drop existing AST rows for this modality so re-runs are idempotent.
        drop_mask = (existing["modality"] == modality) & (existing["model"] == "ast")
        existing = existing[~drop_mask]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined = combined[VOLUMETRICS_COLUMNS]
    combined.to_csv(VOLUMETRICS_CSV, index=False)
    print(f"[wrote] {VOLUMETRICS_CSV} ({len(combined)} rows)")
    return VOLUMETRICS_CSV


def _stable_figure_names(modality, row):
    """Rename per-experiment PNGs to the canonical ``..._ast.png`` flat names."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    src_curve = row.get("learning_curve_png") or row.get("curve_png")
    dst_curve = os.path.join(FIGURES_DIR, f"learning_curve_{modality}_ast.png")
    if src_curve and os.path.exists(src_curve) and os.path.abspath(src_curve) != os.path.abspath(dst_curve):
        os.replace(src_curve, dst_curve)
    row["learning_curve_png"] = dst_curve

    src_cm = row.get("cm_figure_path")
    dst_cm = os.path.join(FIGURES_DIR, f"cm_{modality}_ast.png")
    if src_cm and os.path.exists(src_cm) and os.path.abspath(src_cm) != os.path.abspath(dst_cm):
        os.replace(src_cm, dst_cm)
    row["cm_figure_path"] = dst_cm
    row["cm_figure"] = os.path.basename(dst_cm)
    return row


def _is_nonconverged(modality, row):
    """True when the AST run did not beat chance."""
    bvs = row.get("best_val_score", None)
    if bvs is None:
        return False
    return float(bvs) <= _CHANCE_THRESHOLD.get(modality, 0.5)


def _run_ast_modality(cache, modality, lr, wall_cap_s, batch_size=32, seed=42):
    """Fine-tune AST on ``modality`` and return the metric row dict.

    The AST input adaptation happens inside ``ASTWrapper.forward``, so the DataLoader
    returns the same ``(B, 1, 64, 128)`` batches as the small-CNN path and the wrapper
    handles the freq/time reshaping internally.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    loaders = build_loaders(
        cache, modality, for_effnet=False, batch_size=batch_size, seed=seed,
    )
    n_classes = loaders["n_classes"]

    # Defence-in-depth leakage guard before training.
    split = np.asarray(cache["split"])
    pid = np.asarray(cache["patient_id"])
    assert_no_patient_leakage(pid[split == "train"], pid[split == "test"])

    rec_id = np.asarray(cache["recording_id"])
    is_tr, is_te = split == "train", split == "test"
    volumetrics = {
        "n_train_segments": int(is_tr.sum()),
        "n_test_segments": int(is_te.sum()),
        "n_train_recordings": len(set(map(str, rec_id[is_tr]))),
        "n_test_recordings": len(set(map(str, rec_id[is_te]))),
        "n_train_patients": len(set(map(str, pid[is_tr]))),
        "n_test_patients": len(set(map(str, pid[is_te]))),
    }

    # Build AST model (downloads / loads from HF cache on first call).
    print(f"  [run_ast] loading AST checkpoint (MIT/ast-finetuned-audioset-10-10-0.4593) ...")
    net = build_ast(n_classes)
    params_count = count_params(net)
    print(f"  [run_ast] AST params={params_count:,}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [run_ast] device={dev}")

    # Weighted cross-entropy using train-only class weights.
    criterion = nn.CrossEntropyLoss(weight=loaders["class_weights"].to(dev))

    val_metric_fn = _val_macc if modality == "heart" else _val_icbhi

    out_dir = os.path.join(FIGURES_DIR, f"{modality}_ast")
    os.makedirs(out_dir, exist_ok=True)

    curve_png = os.path.join(out_dir, f"learning_curve_{modality}_ast.png")
    ckpt_path = os.path.join(out_dir, f"ckpt_{modality}_ast.pt")

    net, train_info = train_one_model(
        net,
        loaders["train_loader"],
        loaders["val_loader"],
        criterion,
        lr=lr,
        val_metric_fn=val_metric_fn,
        max_epochs=30,
        patience=10,  # larger model converges slower than the small CNN
        wall_cap_s=wall_cap_s,
        ckpt_path=ckpt_path,
        curve_png=curve_png,
        device=dev,
    )

    row = evaluate(
        net,
        loaders["test_loader"],
        modality,
        loaders["test_recording_id"],
        out_dir,
        _AST_FS_LABEL,
        "ast",
        params_count,
        train_info["train_time_s"],
        volumetrics,
        device=dev,
    )

    row["best_val_score"] = train_info["best_val_score"]
    row["epochs_ran"] = train_info["epochs_ran"]
    row["lr"] = float(lr)
    row["curve_png"] = curve_png
    row["learning_curve_png"] = curve_png
    row["ckpt_path"] = ckpt_path
    row["fallback_from"] = ""
    row["feature_set"] = _AST_FS_LABEL

    return row


def run_modality(modality, lr, wall_cap_s):
    """Run the AST fine-tune for ``modality`` and write all CSVs + figures.

    On any failure (download, OOM, non-convergence) it records a limitation row in
    ``metrics_ast.csv`` with a NaN primary metric instead of crashing; that row is not
    merged into ``unified_comparison.csv``.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cache, cache_path = _load_cache(modality)

    # Re-assert no patient leakage at startup (logs the [leakage-check OK] line).
    pid = np.asarray(list(map(str, cache["patient_id"])), dtype=object)
    split = np.asarray(cache["split"], dtype=object)
    assert_no_patient_leakage(pid[split == "train"], pid[split == "test"])

    print(f"[run_ast] modality={modality} lr={lr} wall_cap_s={wall_cap_s} cache={cache_path}")

    try:
        row = _run_ast_modality(cache, modality, lr=lr, wall_cap_s=wall_cap_s)

        if _is_nonconverged(modality, row):
            bvs = row.get("best_val_score", float("nan"))
            print(
                f"  [run_ast] AST non-converged (best_val_score={bvs:.4f} <= chance) — "
                f"recording as limitation; skipping unified_comparison.csv merge."
            )
            row["fallback_from"] = "non_converged"
            row["primary_metric"] = float("nan")
            _write_metrics_csv(modality, [row])
            print(
                f"  [run_ast] AST {modality} recorded as limitation (non_converged) in "
                f"metrics_ast.csv. Report under Limitations section."
            )
            return []

        # Success: normalise figure names to the canonical flat names.
        row = _stable_figure_names(modality, row)

        print(
            f"  [run_ast] AST {modality}: best_val_score={row.get('best_val_score'):.4f} "
            f"primary={row.get('primary_metric'):.4f} "
            f"epochs={row.get('epochs_ran')} params={row.get('params')}"
        )

        _write_metrics_csv(modality, [row])
        _rebuild_unified(modality, [row])
        _rebuild_volumetrics(modality, [row], cache_path)
        return [row]

    except Exception as exc:
        # Catch download failures, OOM, and any other error; classify into a
        # machine-readable code for the fallback_from column.
        exc_str = str(exc).lower()
        if "download" in exc_str or "connection" in exc_str or "http" in exc_str or "hf_" in exc_str:
            failure_code = "hf_download_failed"
        elif "out of memory" in exc_str or "oom" in exc_str or "cuda out" in exc_str:
            failure_code = "oom"
        else:
            failure_code = "non_converged"

        print(
            f"  [run_ast] AST {modality} FAILED ({failure_code}): {exc}\n"
            f"  Recording as honest limitation — skipping unified_comparison.csv merge.\n"
            f"  Add to report Limitations section."
        )

        # Limitation row with a NaN primary metric.
        n_classes = 2 if modality == "heart" else 4
        primary_name = "MAcc" if modality == "heart" else "ICBHI_Score"
        limit_row = {
            "feature_set": _AST_FS_LABEL,
            "model": "ast",
            "primary_metric_name": primary_name,
            "primary_metric": float("nan"),
            primary_name: float("nan"),
            "Se": float("nan"),
            "Sp": float("nan"),
            "macro_f1": float("nan"),
            "auc_roc": "",
            "accuracy": float("nan"),
            "n_train": "",
            "n_test": "",
            "params": "",
            "fallback_from": failure_code,
            "cm_figure": "",
        }
        if modality == "lung":
            for k in ("se_crackle", "se_wheeze", "se_both", "se_normal"):
                limit_row[k] = float("nan")

        _write_metrics_csv(modality, [limit_row])
        print(
            f"  [run_ast] wrote limitation row to metrics_ast.csv "
            f"(fallback_from={failure_code!r})."
        )
        return []


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Fine-tune pretrained AST (MIT/ast-finetuned-audioset-10-10-0.4593) on "
            "heart and/or lung modalities; write metrics_ast.csv + 2 unified rows + figures."
        )
    )
    ap.add_argument("--modality", required=True, choices=["heart", "lung", "all"])
    ap.add_argument(
        "--lr",
        type=float,
        default=_DEFAULT_LR,
        help=f"Fine-tune learning rate (default {_DEFAULT_LR} — lower than CNN's 1e-3).",
    )
    ap.add_argument(
        "--wall-cap-min",
        type=float,
        default=45.0,
        help="Per-modality wall-clock cap in minutes (default 45 — GPU has 4×A100).",
    )
    args = ap.parse_args()

    modalities = ["heart", "lung"] if args.modality == "all" else [args.modality]
    wall_cap_s = int(args.wall_cap_min * 60)

    for m in modalities:
        run_modality(m, lr=args.lr, wall_cap_s=wall_cap_s)
    sys.exit(0)


if __name__ == "__main__":
    main()
