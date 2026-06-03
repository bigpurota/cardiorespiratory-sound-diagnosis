"""Cache classical feature matrices for one modality."""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config

import numpy as np
import pandas as pd

from src.config_loader import load_params
from src.features import extract_features
from src.split import assert_no_patient_leakage

MANIFEST_CSV = os.path.join(config.DATA_PROCESSED, "manifest.csv")
LUNG_CYCLES_CSV = os.path.join(config.DATA_PROCESSED, "lung_cycles.csv")
HEART_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "heart_splits.csv")
LUNG_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "lung_splits.csv")
FEATURES_DIR = os.path.join(config.PROJECT_ROOT, "features")


def _load_inputs(modality):
    """Return (df, splits_df) for the modality, with patient_id"""
    if modality == "heart":
        df = pd.read_csv(MANIFEST_CSV)
        df = df[df.modality == "heart"].copy()
        df["patient_id"] = df["patient_id"].astype(str)
        splits = pd.read_csv(HEART_SPLITS_CSV)
    else:
        df = pd.read_csv(LUNG_CYCLES_CSV)
        df["patient_id"] = df["patient_id"].astype(str)
        splits = pd.read_csv(LUNG_SPLITS_CSV)
    splits["patient_id"] = splits["patient_id"].astype(str)
    splits["split"] = splits["split"].astype(str)
    return df, splits


def build(modality):
    """Build and cache the feature matrix for ``modality``; print"""
    df, splits = _load_inputs(modality)

    train_ids = splits.loc[splits.split == "train", "patient_id"]
    test_ids = splits.loc[splits.split == "test", "patient_id"]
    assert_no_patient_leakage(train_ids, test_ids)

    params = load_params(modality)
    payload = extract_features(modality, df, splits, params, include_spectral_both=True)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_path = os.path.join(FEATURES_DIR, f"{modality}_classical.npy")
    np.save(out_path, payload, allow_pickle=True)

    split_arr = payload["split"]
    rec_arr = payload["recording_id"]
    pid_arr = payload["patient_id"]
    n_train = int((split_arr == "train").sum())
    n_test = int((split_arr == "test").sum())
    unit = "windows" if modality == "heart" else "cycles"
    size_mb = os.path.getsize(out_path) / 1e6
    print(
        f"[build OK] modality={modality} cache={out_path}\n"
        f"  X_A={payload['X_A'].shape} X_B={payload['X_B'].shape} "
        f"labels={payload['labels'].shape[0]}\n"
        f"  {unit}: train={n_train} test={n_test} total={n_train + n_test}\n"
        f"  recordings={len(set(map(str, rec_arr)))} patients={len(set(map(str, pid_arr)))}\n"
        f"  data_volume_mb={size_mb:.2f}"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build & cache classical feature matrices.")
    ap.add_argument("--modality", required=True, choices=["heart", "lung"])
    args = ap.parse_args()
    build(args.modality)
    sys.exit(0)


if __name__ == "__main__":
    main()
