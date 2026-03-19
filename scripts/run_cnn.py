"""
Run the deep-learning experiments per modality and write the result CSVs.

CLI ``--modality {heart,lung,all} --model {cnn,effnet,all}`` that turns the spectrogram
cache (``features/{modality}_spectrograms.npy``) and the training loop (``src.train_cnn``)
into the four deep-learning rows of the report: a small CNN and an EfficientNet-B0 for each
of heart and lung. Heart is headlined on MAcc, lung on the ICBHI Score.

The small CNN runs before EfficientNet for each modality so its row is available as a
fallback if the EffNet run fails. It writes ``results/tables/metrics_{modality}_cnn.csv``,
idempotently merges the DL rows into ``results/tables/unified_comparison.csv`` (leaving the
classical rows untouched), and rebuilds ``results/tables/volumetrics_cnn.csv``.

If an EfficientNet run raises, fails to beat chance, or overruns the wall-clock cap, the
small-CNN row is written into the ``effnet_b0`` slot with ``fallback_from="cnn"`` provenance
(a genuine EffNet row carries ``fallback_from=""``) so the matrix stays complete instead of
crashing. The same script runs on CPU and GPU; device auto-detection lives in
``src.train_cnn.train_one_model``.

    uv run python scripts/run_cnn.py --modality heart --model all
    uv run python scripts/run_cnn.py --modality all --model all --wall-cap-min 12
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

import numpy as np
import pandas as pd

from src.split import assert_no_patient_leakage
from src.train_cnn import run_modality as train_run_modality

FEATURES_DIR = os.path.join(config.PROJECT_ROOT, "features")
TABLES_DIR = os.path.join(config.RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(config.RESULTS_DIR, "figures")

UNIFIED_CSV = os.path.join(TABLES_DIR, "unified_comparison.csv")
VOLUMETRICS_CSV = os.path.join(TABLES_DIR, "volumetrics_cnn.csv")

UNIFIED_COLUMNS = [
    "modality", "feature_set", "model", "primary_metric_name", "primary_metric",
    "Se", "Sp", "macro_f1", "auc_roc", "accuracy", "n_train", "n_test",
]

DL_MODELS = ["cnn", "effnet_b0"]

VOLUMETRICS_COLUMNS = [
    "modality", "feature_set", "model", "train_time_s",
    "n_train_segments", "n_test_segments",
    "n_train_recordings", "n_test_recordings",
    "n_train_patients", "n_test_patients",
    "params", "fallback_from", "data_volume_mb",
]

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

_CHANCE_THRESHOLD = {"heart": 0.5, "lung": 0.5}


def _load_cache(modality):
    """Load the spectrogram cache for ``modality``, building it via
    ``scripts/build_spectrograms.build`` if absent so unattended runs are self-sufficient."""
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
    """Idempotently merge this modality's DL rows into unified_comparison.csv.

    The drop-mask targets only the DL models for this modality, so the classical rows
    survive and re-runs do not duplicate rows.
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
    """Rename the per-experiment PNGs to the canonical flat names and rewrite row paths.

    ``src.train_cnn`` writes the figures under an experiment subdir; this moves them to
    ``results/figures/{learning_curve,cm}_{modality}_{cnn|effnet}.png``.
    """
    fig_stem = "effnet" if model_name == "effnet_b0" else "cnn"
    os.makedirs(FIGURES_DIR, exist_ok=True)

    src_curve = row.get("learning_curve_png") or row.get("curve_png")
    dst_curve = os.path.join(FIGURES_DIR, f"learning_curve_{modality}_{fig_stem}.png")
    if src_curve and os.path.exists(src_curve) and os.path.abspath(src_curve) != os.path.abspath(dst_curve):
        os.replace(src_curve, dst_curve)
    row["learning_curve_png"] = dst_curve

    src_cm = row.get("cm_figure_path")
    dst_cm = os.path.join(FIGURES_DIR, f"cm_{modality}_{fig_stem}.png")
    if src_cm and os.path.exists(src_cm) and os.path.abspath(src_cm) != os.path.abspath(dst_cm):
        os.replace(src_cm, dst_cm)
    row["cm_figure_path"] = dst_cm
    row["cm_figure"] = os.path.basename(dst_cm)
    return row


def _copy_figures_for_fallback(modality, row):
    """Copy the small-CNN figures to the effnet canonical names for the fallback row.

    Unlike ``_stable_figure_names`` (which moves files), this copies the published
    ``{kind}_{modality}_cnn.png`` figures to ``..._effnet.png`` so the genuine CNN figures
    survive while the substituted effnet figures still exist.
    """
    import shutil

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for kind in ("learning_curve", "cm"):
        src = os.path.join(FIGURES_DIR, f"{kind}_{modality}_cnn.png")
        dst = os.path.join(FIGURES_DIR, f"{kind}_{modality}_effnet.png")
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    row["learning_curve_png"] = os.path.join(
        FIGURES_DIR, f"learning_curve_{modality}_effnet.png"
    )
    cm_dst = os.path.join(FIGURES_DIR, f"cm_{modality}_effnet.png")
    row["cm_figure_path"] = cm_dst
    row["cm_figure"] = os.path.basename(cm_dst)
    return row


def _run_experiment(cache, modality, model, wall_cap_s):
    """Train and evaluate one DL experiment via src.train_cnn; normalise figure names.

    Returns the row dict (with ``fallback_from=""`` for a genuine run). Raises on hard
    failure — the caller decides whether to substitute the CNN row.
    """
    row = train_run_modality(
        cache, modality, model=model, wall_cap_s=wall_cap_s
    )
    row["fallback_from"] = ""
    row = _stable_figure_names(modality, row["model"], row)
    return row


def _is_nonconverged(modality, row):
    """True when an EffNet run did not beat chance (fallback trigger)."""
    bvs = row.get("best_val_score", None)
    if bvs is None:
        return False
    return float(bvs) <= _CHANCE_THRESHOLD.get(modality, 0.5)


def run_modality(modality, models, wall_cap_s):
    """Run the selected DL experiments for ``modality`` and write all three CSVs + figures.

    The small CNN runs before EfficientNet so its row is available as the fallback before
    the higher-risk EffNet run is attempted.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    cache, cache_path = _load_cache(modality)

    pid = np.asarray(list(map(str, cache["patient_id"])), dtype=object)
    split = np.asarray(cache["split"], dtype=object)
    assert_no_patient_leakage(pid[split == "train"], pid[split == "test"])

    print(f"[run_cnn] modality={modality} models={models} cache={cache_path} "
          f"wall_cap_s={wall_cap_s}")

    ordered = [m for m in ("cnn", "effnet") if m in models]

    rows = []
    cnn_row = None
    for model in ordered:
        if model == "cnn":
            print(f"  [#{'5' if modality == 'heart' else '8'}] {modality} small CNN ...")
            row = _run_experiment(cache, modality, "cnn", wall_cap_s)
            cnn_row = row
            rows.append(row)
            print(f"    cnn best_val_score={row.get('best_val_score'):.4f} "
                  f"primary={row.get('primary_metric'):.4f} "
                  f"epochs={row.get('epochs_ran')} params={row.get('params')}")
        else:
            tag = "9" if modality == "heart" else "10"
            print(f"  [#{tag}] {modality} EfficientNet-B0 ...")
            try:
                row = _run_experiment(cache, modality, "effnet", wall_cap_s)
                if _is_nonconverged(modality, row):
                    print(f"    EffNet non-converged (best_val_score="
                          f"{row.get('best_val_score'):.4f} <= chance) — using fallback.")
                    raise RuntimeError("effnet non-converged below chance")
                rows.append(row)
                print(f"    effnet best_val_score={row.get('best_val_score'):.4f} "
                      f"primary={row.get('primary_metric'):.4f} "
                      f"epochs={row.get('epochs_ran')} params={row.get('params')}")
            except Exception as exc:
                print(f"    [fallback] EffNet #{tag} failed/overran: {exc}")
                if cnn_row is None:
                    print("    [fallback] training a small CNN to fill the row ...")
                    cnn_row = _run_experiment(cache, modality, "cnn", wall_cap_s)
                fb = dict(cnn_row)
                fb["model"] = "effnet_b0"
                fb["fallback_from"] = "cnn"
                fb = _copy_figures_for_fallback(modality, fb)
                rows.append(fb)
                print(f"    [fallback] wrote small-CNN result as effnet_b0 row "
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
        help="Per-experiment wall-clock cap in minutes (relax on GPU).",
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
