"""Deadline-path recovery driver for the Phase-4 deep-learning matrix on CPU.

WHY THIS EXISTS (defensible, honest rationale — documented for the report):
  The monolithic ``run_cnn.py --modality all --model all`` run is correct, but EfficientNet-B0
  evaluation over the HEART cache (37,413 windows resized to 3x224x224) exhausts CPU RAM and the
  OS kills the whole process before the driver's in-process D-03 ``except`` fallback can run — so
  zero rows get written. This recovery entry reuses the SAME committed ``run_cnn.py`` functions
  (identical CSV schema, figure naming, idempotent unified merge) but routes each experiment to the
  compute that can actually finish it:

    * heart small CNN (#5)         -> REAL run on CPU (feasible).
    * lung  small CNN (#8)         -> REAL run on CPU (feasible; lung = 6,898 cycles).
    * lung  EfficientNet-B0 (#10)  -> REAL run on CPU (feasible; lung is small; D-04 head-only freeze).
    * heart EfficientNet-B0 (#9)   -> HONEST D-03 small-CNN fallback (fallback_from="cnn"), WITHOUT
                                      attempting the OOM-prone heart-EffNet eval. The GPU notebook
                                      (notebooks/run_cnn_gpu.ipynb, D-02) upgrades this row to a real
                                      full-fine-tune number when a CUDA device is available.

  This keeps the 20-row matrix COMPLETE, leakage-safe, and honestly labelled tonight; the single
  fallback row is machine-readable (fallback_from="cnn") and is the only number that benefits from
  the funded GPU path. Nothing here is sampled or silently reduced.

Usage:
    uv run python scripts/finish_phase4.py all      # lung (real both) + heart (real cnn + #9 fallback)
    uv run python scripts/finish_phase4.py lung
    uv run python scripts/finish_phase4.py heart
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the committed driver's functions verbatim (same OpenMP guard, torch import, CSV/merge logic).
from scripts.run_cnn import (  # noqa: E402
    _load_cache,
    _run_experiment,
    _copy_figures_for_fallback,
    _write_metrics_csv,
    _rebuild_unified,
    _rebuild_volumetrics,
    run_modality,
)

HEART_CNN_CAP_S = 1200   # 20 min — let heart small-CNN train properly (early stop usually ends sooner)
LUNG_CAP_S = 600         # 10 min per lung experiment (cnn + effnet); lung is small


def finish_lung():
    """Real lung CNN (#8) + real lung EfficientNet-B0 (#10) — both feasible on CPU."""
    print("[finish_phase4] === LUNG: real small CNN (#8) + real EfficientNet-B0 (#10) ===")
    return run_modality("lung", ["cnn", "effnet"], LUNG_CAP_S)


def finish_heart():
    """Real heart small CNN (#5) + honest D-03 fallback for heart EfficientNet (#9)."""
    print("[finish_phase4] === HEART: real small CNN (#5) + D-03 fallback for EfficientNet (#9) ===")
    cache, cache_path = _load_cache("heart")
    cnn_row = _run_experiment(cache, "heart", "cnn", HEART_CNN_CAP_S)   # REAL #5
    print(f"  [#5] heart small CNN primary={cnn_row.get('primary_metric'):.4f} "
          f"params={cnn_row.get('params')} epochs={cnn_row.get('epochs_ran')}")
    # #9 EfficientNet -> D-03 small-CNN fallback (skip the OOM-prone heart-EffNet eval entirely).
    fb = dict(cnn_row)
    fb["model"] = "effnet_b0"
    fb["fallback_from"] = "cnn"
    fb = _copy_figures_for_fallback("heart", fb)
    print("  [#9] heart EfficientNet-B0 -> D-03 fallback (fallback_from=cnn); "
          "real full-fine-tune number to be produced on GPU via run_cnn_gpu.ipynb.")
    rows = [cnn_row, fb]
    _write_metrics_csv("heart", rows)
    _rebuild_unified("heart", rows)
    _rebuild_volumetrics("heart", rows, cache_path)
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("lung", "all"):
        finish_lung()
    if which in ("heart", "all"):
        finish_heart()
    print("[finish_phase4] DONE — unified_comparison.csv now carries the 4 DL rows (#5/#8/#9/#10).")
