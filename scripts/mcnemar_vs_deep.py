"""Paired McNemar between the best classical heart model (XGBoost-B) and the deep
EfficientNet-B0, using per-recording predictions.

The deep predictions are produced by running the retained in-domain EfficientNet-B0
checkpoints (three seeds) through test inference on the GPU box and exported to
results/tables/effnet_heart_preds.csv. The classical predictions are refit locally
with the exact Chapter 3 pipeline. Both share the same recording-level test split, so
the two correctness vectors pair recording-for-recording. Deterministic, CPU.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

from src import config
from src.metrics import majority_vote, heart_macc
from scripts.mcnemar_classical import _fit_tuned_serial, mcnemar

ROOT = os.path.dirname(config.RESULTS_DIR)
HEART = os.path.join(ROOT, "features", "heart_classical.npy")
EFFNET = os.path.join(config.RESULTS_DIR, "tables", "effnet_heart_preds.csv")
OUT = os.path.join(config.RESULTS_DIR, "tables", "mcnemar_vs_deep.csv")
SEEDS = (1, 2, 42)


def xgb_per_recording():
    cache = np.load(HEART, allow_pickle=True).item()
    split = np.asarray(cache["split"], dtype=object)
    labels = np.asarray(cache["labels"], dtype=int)
    pid = np.asarray(list(map(str, cache["patient_id"])), dtype=object)
    rec = np.asarray(list(map(str, cache["recording_id"])), dtype=object)
    tr, te = split == "train", split == "test"
    X = np.asarray(cache["X_B"], dtype="float64")
    est = _fit_tuned_serial("xgb", 2, X[tr], labels[tr], pid[tr], seed=42)
    pred = est.predict(X[te])
    pred_rec = majority_vote(pred, rec[te])
    true_rec = majority_vote(labels[te], rec[te]).reindex(pred_rec.index)
    macc = heart_macc(true_rec.to_numpy().astype(int), pred_rec.to_numpy().astype(int))["MAcc"]
    return pd.DataFrame({
        "recording_id": pred_rec.index.astype(str),
        "true": true_rec.to_numpy().astype(int),
        "xgb": pred_rec.to_numpy().astype(int),
    }), float(macc)


def run():
    xgb_df, xgb_macc = xgb_per_recording()
    eff = pd.read_csv(EFFNET).astype({"recording_id": str})
    df = xgb_df.merge(eff.drop(columns=["true"]), on="recording_id")
    print(f"XGBoost-B MAcc={xgb_macc:.4f}  (report 0.9025);  paired on {len(df)} recordings\n")

    xgb_correct = (df.xgb.to_numpy() == df.true.to_numpy())
    rows = []
    for s in SEEDS:
        eff_correct = (df[f"effnet_s{s}"].to_numpy() == df.true.to_numpy())
        eff_macc = heart_macc(df.true.to_numpy(), df[f"effnet_s{s}"].to_numpy())["MAcc"]
        xgb_only, eff_only, stat, pe, pc = mcnemar(xgb_correct, eff_correct)
        verdict = "DIFFER (p<0.05)" if pe < 0.05 else "no evidence of difference"
        print(f"seed {s}: EffNet MAcc={eff_macc:.4f} | discordant XGB-only={xgb_only} "
              f"EffNet-only={eff_only} | chi2={stat:.3f} p_exact={pe:.4f} -> {verdict}")
        rows.append(dict(seed=s, xgb_macc=round(xgb_macc, 4), effnet_macc=round(float(eff_macc), 4),
                         n=len(df), xgb_only_right=xgb_only, effnet_only_right=eff_only,
                         chi2_cc=round(stat, 3), p_exact=round(pe, 4), p_chi=round(pc, 4)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
