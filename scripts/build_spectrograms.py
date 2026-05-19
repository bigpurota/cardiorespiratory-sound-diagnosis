"""
Cache log-mel spectrograms for one modality (input to the CNN models).

Closely parallels scripts/build_features.py — same split join, startup leakage
check, payload format and volumetric print — but replaces classical feature
extraction with the mel transform. The audio path (load_resampled → bandpass_sos
→ peak_normalize) is identical to src/features.extract_features so the deep-learning
rows stay comparable with the classical ones.

Output is features/{modality}_spectrograms.npy holding X (N×64×128 float32) plus
labels, patient_id, split and recording_id. No augmentation happens here; spec-augment
and noise live only in the train DataLoader. Labels come from the manifest/cycles CSV
via HEART_LABEL_MAP / LUNG_LABEL_MAP, never from filenames.

    uv run python scripts/build_spectrograms.py --modality heart
    uv run python scripts/build_spectrograms.py --modality lung
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config  # noqa: F401  import FIRST (seeds RNGs, exposes paths)

import numpy as np
import pandas as pd

from src.config_loader import load_params
from src.features import HEART_LABEL_MAP, LUNG_LABEL_MAP
from src.preprocess import load_resampled, bandpass_sos, peak_normalize
from src.segment import segment_fixed
from src.spectrograms import make_mel, window_to_logmel
from src.split import assert_no_patient_leakage

MANIFEST_CSV = os.path.join(config.DATA_PROCESSED, "manifest.csv")
LUNG_CYCLES_CSV = os.path.join(config.DATA_PROCESSED, "lung_cycles.csv")
HEART_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "heart_splits.csv")
LUNG_SPLITS_CSV = os.path.join(config.SPLITS_DIR, "lung_splits.csv")
FEATURES_DIR = os.path.join(config.PROJECT_ROOT, "features")

WINDOW_SAMPLES = 12000  # 3.0 s @ 4000 Hz — fixed log-mel input length


def _load_inputs(modality):
    """Return (df, splits_df) for the modality, with patient_id as str.

    Same logic as scripts/build_features.py: heart reads the manifest filtered to
    ``modality == "heart"``, lung reads lung_cycles.csv, and both join the split CSV.
    """
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


def _extract_spectrograms(modality, df, splits, params):
    """Run the audio path + mel transform into aligned (X, labels, pid, split, rec_id).

    Follows src/features.extract_features but emits a (64,128) log-mel per window/cycle
    instead of a feature vector. The mel filterbank is built once per modality (fmin/fmax
    from the params bandpass) and reused for every row.
    """
    fmin = int(params.get("bandpass_low_hz"))
    fmax = int(params.get("bandpass_high_hz"))
    order = int(params.get("bandpass_order", 4))
    sr = int(params.get("sample_rate", 4000))

    mel = make_mel(fmin, fmax, sr=sr)  # built once, reused for every row

    split_lookup = {str(r.patient_id): str(r.split) for r in splits.itertuples()}

    X_rows, labels, pids, splits_out, rec_ids = [], [], [], [], []

    if modality == "heart":
        for row in df.itertuples():
            pid = str(row.patient_id)
            split = split_lookup.get(pid)
            if split is None:
                continue  # recording not in the split
            label = HEART_LABEL_MAP.get(float(row.label))
            if label is None:
                continue  # unsure / out-of-domain label
            y = load_resampled(row.filepath, target_sr=sr)
            yb = peak_normalize(bandpass_sos(y, fmin, fmax, fs=sr, order=order))
            # 50% overlap on train, none on test (same as the classical path).
            hop_s = 1.5 if split == "train" else 3.0
            for w in segment_fixed(yb, win_s=3.0, hop_s=hop_s, fs=sr):
                spec = window_to_logmel(w, mel)
                X_rows.append(spec)
                labels.append(label)
                pids.append(pid)
                splits_out.append(split)
                rec_ids.append(pid)  # heart recording_id == patient_id
    elif modality == "lung":
        # Preprocess each recording once, then slice all of its cycles.
        for filepath, grp in df.groupby("filepath"):
            first = grp.iloc[0]
            pid = str(first.patient_id)
            split = split_lookup.get(pid)
            if split is None:
                continue
            rec_id = pathlib.Path(str(filepath)).stem
            y = load_resampled(filepath, target_sr=sr)
            yb = peak_normalize(bandpass_sos(y, fmin, fmax, fs=sr, order=order))
            for cyc in grp.itertuples():
                label = LUNG_LABEL_MAP.get(str(cyc.label).strip().lower())
                if label is None:
                    continue  # unknown label string
                s, e = int(float(cyc.start_s) * sr), int(float(cyc.end_s) * sr)
                clip = yb[s:e]
                # Pad/trim to exactly 12000 samples, matching lung_cycle_vector.
                if len(clip) < WINDOW_SAMPLES:
                    clip = np.pad(clip, (0, WINDOW_SAMPLES - len(clip)))
                else:
                    clip = clip[:WINDOW_SAMPLES]
                spec = window_to_logmel(clip, mel)
                X_rows.append(spec)
                labels.append(label)
                pids.append(pid)
                splits_out.append(split)
                rec_ids.append(rec_id)
    else:
        raise ValueError(f"Unknown modality '{modality}'. Expected 'heart' or 'lung'.")

    X = (
        np.asarray(X_rows, dtype="float32")
        if X_rows
        else np.empty((0, 64, 128), dtype="float32")
    )
    assert np.all(np.isfinite(X)), "cached spectrogram tensor contains NaN/Inf"
    return {
        "X": X,
        "labels": np.asarray(labels, dtype=int),
        "patient_id": np.asarray(pids, dtype=object),
        "split": np.asarray(splits_out, dtype=object),
        "recording_id": np.asarray(rec_ids, dtype=object),
    }


def build(modality):
    """Build and cache the spectrogram tensor for ``modality``; print volumetrics."""
    df, splits = _load_inputs(modality)

    # Re-check patient-level separation before extracting anything.
    train_ids = splits.loc[splits.split == "train", "patient_id"]
    test_ids = splits.loc[splits.split == "test", "patient_id"]
    assert_no_patient_leakage(train_ids, test_ids)

    params = load_params(modality)
    payload = _extract_spectrograms(modality, df, splits, params)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_path = os.path.join(FEATURES_DIR, f"{modality}_spectrograms.npy")
    np.save(out_path, payload, allow_pickle=True)

    # Report window/cycle counts, recordings and patients.
    split_arr = payload["split"]
    rec_arr = payload["recording_id"]
    pid_arr = payload["patient_id"]
    n_train = int((split_arr == "train").sum())
    n_test = int((split_arr == "test").sum())
    unit = "windows" if modality == "heart" else "cycles"
    size_mb = os.path.getsize(out_path) / 1e6
    print(
        f"[build OK] modality={modality} cache={out_path}\n"
        f"  X={payload['X'].shape} dtype={payload['X'].dtype} "
        f"labels={payload['labels'].shape[0]}\n"
        f"  {unit}: train={n_train} test={n_test} total={n_train + n_test}\n"
        f"  recordings={len(set(map(str, rec_arr)))} patients={len(set(map(str, pid_arr)))}\n"
        f"  data_volume_mb={size_mb:.2f}"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Build & cache log-mel spectrogram tensors.")
    ap.add_argument("--modality", required=True, choices=["heart", "lung"])
    args = ap.parse_args()
    build(args.modality)
    sys.exit(0)


if __name__ == "__main__":
    main()
