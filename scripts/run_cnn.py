#!/usr/bin/env python
"""
scripts/run_cnn.py — run the 4 deep-learning experiments per modality & write CSVs.

EXACT analog of ``scripts/run_classical.py`` (the classical driver): a CLI
``--modality {heart,lung,all} --model {cnn,effnet,all}`` that turns the Wave-2 spectrogram
cache (``features/{modality}_spectrograms.npy``) and the Wave-3 models/training loop
(``src.train_cnn``) into the 4 deep-learning rows of the report:

  - #5  heart small CNN   (model=cnn,       modality=heart)
  - #8  lung  small CNN   (model=cnn,       modality=lung)
  - #9  heart EfficientNet (model=effnet_b0, modality=heart)
  - #10 lung  EfficientNet (model=effnet_b0, modality=lung)

CNNs (#5/#8) are sequenced BEFORE EfficientNet (#9/#10) so the lung small-CNN row is in
hand as the D-03 fallback before the high-risk #10 EffNet-ICBHI run is attempted
(Open Question 2 / 04-RESEARCH §Open Questions).

The driver:
  1. loads / builds the spectrogram cache (``features/{modality}_spectrograms.npy``; if
     absent it invokes ``scripts/build_spectrograms.build`` — same payload schema);
  2. re-asserts ``assert_no_patient_leakage`` on the cached train/test patient groups
     (D-03 — logs the ``[leakage-check OK]`` line);
  3. for each (modality, model) selected, calls ``src.train_cnn.run_modality`` to
     build leakage-safe loaders → train (early stop + wall cap) → evaluate, writing a
     learning-curve PNG + confusion-matrix PNG + best checkpoint;
  4. writes ``results/tables/metrics_{modality}_cnn.csv`` (heart headlined MAcc / lung
     headlined ICBHI_Score, full suite + per-class Se for lung + ``params`` +
     ``fallback_from`` provenance), idempotently merges the DL rows into
     ``results/tables/unified_comparison.csv`` (drop-mask targets model∈{cnn,effnet_b0}
     so the 16 classical rows survive → 20 long-format rows), and rebuilds
     ``results/tables/volumetrics_cnn.csv`` (Pattern-9 + a ``params`` column).

D-03 EffNet→small-CNN fallback hook: if an EfficientNet run raises, fails to converge
(best_val_score not better than chance), or overruns the wall-clock cap, the corresponding
small-CNN row is written as the ``effnet_b0`` row WITH a machine-readable
``fallback_from="cnn"`` provenance value (a genuine EffNet row carries ``fallback_from=""``)
— an honest partial matrix instead of a crash. The final accept/keep decision is the
Plan-04 Task-3 human checkpoint.

The SAME script runs on CPU (fallback path; EffNet head-only freeze per D-04) and on the
funded GPU (primary; full EffNet fine-tune) — device auto-detect lives in
``src.train_cnn.train_one_model`` so there is no code fork (D-07).

    uv run python scripts/run_cnn.py --modality heart --model all
    uv run python scripts/run_cnn.py --modality all --model all --wall-cap-min 12
"""
import os

# macOS duplicate-OpenMP-runtime guard (copied VERBATIM from src/train_cnn.py): torch and
# any sklearn/xgboost loaded alongside each bundle their own libomp.dylib; capping OpenMP
# to a single team + allowing the duplicate runtime prevents the collision segfault. MUST
# run BEFORE the first ``import torch`` (transitively pulled in below). ``setdefault`` lets a
# caller override.
os.environ.setdefault("OMP_NUM_THREADS", "1")
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

from src.split import assert_no_patient_leakage  # noqa: E402
from src.train_cnn import run_modality as train_run_modality  # noqa: E402

FEATURES_DIR = os.path.join(config.PROJECT_ROOT, "features")
TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(config.RESULTS_DIR, "figures")

UNIFIED_CSV = os.path.join(TABLES_DIR, "unified_comparison.csv")
VOLUMETRICS_CSV = os.path.join(TABLES_DIR, "volumetrics_cnn.csv")

# Pattern-8 unified_comparison.csv column order (IDENTICAL to scripts/run_classical.py —
# the 12-column schema DL rows MUST match; never add/reorder columns, T-04-08).
UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]

# DL models written by this driver (the drop-mask target — NOT the classical MODEL_NAMES).
DL_MODELS = ["cnn", "effnet_b0"]

# Pattern-9 volumetrics_cnn.csv column order — mirrors volumetrics_classical.csv + a
# ``params`` column (D-09 DL volumetric field).
VOLUMETRICS_COLUMNS = [
    "modality", "feature_set", "model", "train_time_s",
    "n_train_segments", "n_test_segments",
    "n_train_recordings", "n_test_recordings",
    "n_train_patients", "n_test_patients",
    "params", "fallback_from", "data_volume_mb",
]

# Per-modality metrics CSV columns (full suite; primary metric is the headline column).
# DL extras vs classical: ``params``, ``fallback_from`` (machine-readable D-03 provenance).
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

# Chance-level best_val_score below which an EffNet run is judged non-converged (D-03):
# heart binary chance MAcc ~= 0.5; lung 4-class ICBHI Score chance ~= 0.5 (Se+Sp)/2.
_CHANCE_THRESHOLD = {"heart": 0.5, "lung": 0.5}


def _load_cache(modality):
    """np.load the Wave-2 spectrogram cache for ``modality`` (build it if absent).

    Mirrors ``scripts/run_classical.py::_load_cache`` but points at
    ``features/{modality}_spectrograms.npy`` and, rather than only raising, materialises the
    cache via ``scripts/build_spectrograms.build`` so an unattended run is self-sufficient.
    """
    path = os.path.join(FEATURES_DIR, f"{modality}_spectrograms.npy")
    if not os.path.exists(path):
        print(
            f"[run_cnn] spectrogram cache missing: {path} — building it via "
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
    """Write the per-modality DL metrics CSV (2 rows) with the full metric suite."""
    cols = METRICS_COLUMNS[modality]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]
    out = os.path.join(TABLES_DIR, f"metrics_{modality}_cnn.csv")
    df.to_csv(out, index=False)
    print(f"[wrote] {out} ({len(df)} rows)")
    return out


def _rebuild_unified(modality, rows):
    """Deterministically merge this modality's DL rows into unified_comparison.csv.

    Copies the idempotent merge from ``scripts/run_classical.py::_rebuild_unified`` but the
    drop-mask targets the DL models — ``existing["model"].isin(["cnn","effnet_b0"]) &
    (existing["modality"]==modality)`` — so the 16 CLASSICAL rows survive and re-runs are
    idempotent (T-04-08). After both heart and lung DL runs the file holds 16 classical + 4
    DL = 20 long-format rows, rewritten in Pattern-8 column order.
    """
    new = pd.DataFrame([{c: r.get(c, "") for c in UNIFIED_COLUMNS} for r in rows])

    if os.path.exists(UNIFIED_CSV):
        existing = pd.read_csv(UNIFIED_CSV)
        for c in UNIFIED_COLUMNS:
            if c not in existing.columns:
                existing[c] = ""
        existing = existing[UNIFIED_COLUMNS]
        # Drop only this modality's DL rows (keep other modality + all 16 classical rows).
        drop_mask = (
            (existing["modality"] == modality)
            & (existing["model"].isin(["cnn", "effnet_b0"]))
        )
        existing = existing[~drop_mask]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new

    combined = combined[UNIFIED_COLUMNS]
    combined.to_csv(UNIFIED_CSV, index=False)
    n_dl = int(combined["model"].isin(DL_MODELS).sum())
    print(f"[wrote] {UNIFIED_CSV} ({len(combined)} rows; {n_dl} DL)")
    return UNIFIED_CSV


def _rebuild_volumetrics(modality, rows, cache_path):
    """Merge this modality's per-run DL volumetrics into volumetrics_cnn.csv (+params)."""
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


def _stable_figure_names(modality, model_name, row):
    """Rename the per-experiment PNGs to the plan's canonical names and rewrite row paths.

    ``src.train_cnn`` writes ``learning_curve_{modality}_{model_name}.png`` /
    ``cm_{modality}_{fs}_{model_name}.png`` under an experiment subdir; the plan/report
    import the flat names ``results/figures/learning_curve_{modality}_{cnn|effnet}.png`` and
    ``results/figures/cm_{modality}_{cnn|effnet}.png``. ``cnn``→``cnn`` and ``effnet_b0``→
    ``effnet`` for the figure stem (matches files_modified in 04-04-PLAN.md).
    """
    fig_stem = "effnet" if model_name == "effnet_b0" else "cnn"
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Learning curve.
    src_curve = row.get("learning_curve_png") or row.get("curve_png")
    dst_curve = os.path.join(FIGURES_DIR, f"learning_curve_{modality}_{fig_stem}.png")
    if src_curve and os.path.exists(src_curve) and os.path.abspath(src_curve) != os.path.abspath(dst_curve):
        os.replace(src_curve, dst_curve)
    row["learning_curve_png"] = dst_curve

    # Confusion matrix.
    src_cm = row.get("cm_figure_path")
    dst_cm = os.path.join(FIGURES_DIR, f"cm_{modality}_{fig_stem}.png")
    if src_cm and os.path.exists(src_cm) and os.path.abspath(src_cm) != os.path.abspath(dst_cm):
        os.replace(src_cm, dst_cm)
    row["cm_figure_path"] = dst_cm
    row["cm_figure"] = os.path.basename(dst_cm)
    return row


def _run_experiment(cache, modality, model, wall_cap_s):
    """Train+evaluate ONE DL experiment via src.train_cnn; normalise figure names.

    Returns the row dict (with ``fallback_from=""`` set for a genuine run). Raises on hard
    failure — the caller's D-03 hook decides whether to substitute the CNN row.
    """
    row = train_run_modality(
        cache, modality, model=model, wall_cap_s=wall_cap_s
    )
    row["fallback_from"] = ""
    row = _stable_figure_names(modality, row["model"], row)
    return row


def _is_nonconverged(modality, row):
    """True when an EffNet run did not beat chance (D-03 fallback trigger)."""
    bvs = row.get("best_val_score", None)
    if bvs is None:
        return False
    return float(bvs) <= _CHANCE_THRESHOLD.get(modality, 0.5)


def run_modality(modality, models, wall_cap_s):
    """Run the selected DL experiments for ``modality`` and write all three CSVs + figures.

    ORDER: small CNN (#5/#8) FIRST, then EfficientNet (#9/#10) — so the small-CNN row is in
    hand as the D-03 fallback before the high-risk EffNet run is attempted (Open Question 2).
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cache, cache_path = _load_cache(modality)

    # D-03: re-assert zero patient leakage at startup ([leakage-check OK] line).
    pid = np.asarray(list(map(str, cache["patient_id"])), dtype=object)
    split = np.asarray(cache["split"], dtype=object)
    assert_no_patient_leakage(pid[split == "train"], pid[split == "test"])

    print(f"[run_cnn] modality={modality} models={models} cache={cache_path} "
          f"wall_cap_s={wall_cap_s}")

    # Sequence CNN before EffNet (Open Question 2): the small-CNN row is the D-03 fallback.
    ordered = [m for m in ("cnn", "effnet") if m in models]

    rows = []
    cnn_row = None  # cached for the EffNet→small-CNN fallback (D-03).
    for model in ordered:
        if model == "cnn":
            print(f"  [#{'5' if modality == 'heart' else '8'}] {modality} small CNN ...")
            row = _run_experiment(cache, modality, "cnn", wall_cap_s)
            cnn_row = row
            rows.append(row)
            print(f"    cnn best_val_score={row.get('best_val_score'):.4f} "
                  f"primary={row.get('primary_metric'):.4f} "
                  f"epochs={row.get('epochs_ran')} params={row.get('params')}")
        else:  # effnet
            tag = "9" if modality == "heart" else "10"
            print(f"  [#{tag}] {modality} EfficientNet-B0 ...")
            try:
                row = _run_experiment(cache, modality, "effnet", wall_cap_s)
                if _is_nonconverged(modality, row):
                    print(f"    EffNet non-converged (best_val_score="
                          f"{row.get('best_val_score'):.4f} <= chance) — D-03 fallback.")
                    raise RuntimeError("effnet non-converged below chance (D-03)")
                rows.append(row)
                print(f"    effnet best_val_score={row.get('best_val_score'):.4f} "
                      f"primary={row.get('primary_metric'):.4f} "
                      f"epochs={row.get('epochs_ran')} params={row.get('params')}")
            except Exception as exc:  # D-03 EffNet→small-CNN fallback (no crash).
                print(f"    [D-03 fallback] EffNet #{tag} failed/overran: {exc}")
                if cnn_row is None:
                    # No small-CNN row in hand (CNN not selected) — train one now so the
                    # fallback exists honestly rather than crashing the matrix.
                    print("    [D-03 fallback] training a small CNN to fill the row ...")
                    cnn_row = _run_experiment(cache, modality, "cnn", wall_cap_s)
                fb = dict(cnn_row)
                fb["model"] = "effnet_b0"          # report the row under the EffNet slot
                fb["fallback_from"] = "cnn"         # machine-readable provenance (D-03)
                fb = _stable_figure_names(modality, "effnet_b0", fb)
                rows.append(fb)
                print(f"    [D-03 fallback] wrote small-CNN result as effnet_b0 row "
                      f"(fallback_from=cnn).")

    _write_metrics_csv(modality, rows)
    _rebuild_unified(modality, rows)
    _rebuild_volumetrics(modality, rows, cache_path)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Run the 4 deep-learning experiments (2 per modality) and write CSVs."
    )
    ap.add_argument("--modality", required=True, choices=["heart", "lung", "all"])
    ap.add_argument("--model", default="all", choices=["cnn", "effnet", "all"])
    ap.add_argument(
        "--wall-cap-min",
        type=float,
        default=30.0,
        help="Per-experiment wall-clock cap in minutes (D-03; relax on paid GPU).",
    )
    args = ap.parse_args()

    modalities = ["heart", "lung"] if args.modality == "all" else [args.modality]
    models = ["cnn", "effnet"] if args.model == "all" else [args.model]
    wall_cap_s = int(args.wall_cap_min * 60)

    for m in modalities:
        run_modality(m, models, wall_cap_s)
    sys.exit(0)


if __name__ == "__main__":
    main()
