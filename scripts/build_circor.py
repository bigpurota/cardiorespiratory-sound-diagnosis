"""Build CirCor DigiScope 2022 caches for the murmur-detection sub-study.

CirCor is a heart-sound dataset with patient-level murmur labels
(Present / Absent / Unknown) and real patient identifiers with several
recordings per subject. We build a genuinely patient-level split (unlike the
recording-level CinC heart split) and reuse the heart spectrogram + classical
feature path. Output caches use the exact heart cache schema, so they run
through run_modality(..., "heart") and run_experiments("heart", ...) unchanged,
aggregating windows to the recording level for MAcc.

Task: binary murmur detection (Absent=0, Present=1); Unknown is dropped.

Performance: librosa feature extraction is CPU-heavy, so the classical pass is
parallelised across recordings with a process pool, and per-process BLAS/numba
threads are capped at 1 to avoid oversubscription thrash. The spectrogram pass
is cheap and runs single-process. Modes: spec | classical | both.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.preprocess import load_resampled, bandpass_sos, peak_normalize
from src.segment import segment_fixed
from src.spectrograms import make_mel, window_to_logmel
from src.features import window_feature_vector, FEATURE_NAMES_A, FEATURE_NAMES_B
from src.split import assert_no_patient_leakage

CIRCOR_DIR = pathlib.Path(config.DATA_RAW) / "circor"
META_CSV = CIRCOR_DIR / "training_data.csv"
AUDIO_DIR = CIRCOR_DIR / "training_data"
FEATURES_DIR = pathlib.Path(config.PROJECT_ROOT) / "features"
PROCESSED_DIR = pathlib.Path(config.DATA_PROCESSED)
SPLITS_DIR = pathlib.Path(config.SPLITS_DIR)

SR = 4000
BP_LOW, BP_HIGH, BP_ORDER = 20, 400, 4
TEST_FRACTION = 0.30
SEED = 42
MURMUR_MAP = {"Absent": 0, "Present": 1}


def _build_manifest():
    """Parse training_data.csv into recording rows with patient-level split."""
    meta = pd.read_csv(META_CSV)
    meta["Patient ID"] = meta["Patient ID"].astype(str)
    meta = meta[meta["Murmur"].isin(MURMUR_MAP)].copy()
    meta["label"] = meta["Murmur"].map(MURMUR_MAP).astype(int)

    patients = meta[["Patient ID", "label"]].drop_duplicates("Patient ID")
    pat_train, pat_test = train_test_split(
        patients["Patient ID"].to_numpy(),
        test_size=TEST_FRACTION,
        random_state=SEED,
        stratify=patients["label"].to_numpy(),
    )
    split_of = {p: "train" for p in pat_train}
    split_of.update({p: "test" for p in pat_test})

    rows = []
    for r in meta.itertuples(index=False):
        # itertuples can't expose "Patient ID" by name (space); use positional.
        pid = str(r[0])
        label = int(r.label)
        for wav in sorted(AUDIO_DIR.glob(f"{pid}_*.wav")):
            rows.append({
                "filepath": str(wav),
                "patient_id": pid,
                "recording_id": wav.stem,
                "label": label,
                "split": split_of[pid],
            })
    man = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    man.to_csv(PROCESSED_DIR / "circor_manifest.csv", index=False)
    (man[["patient_id", "split"]].drop_duplicates()
        .to_csv(SPLITS_DIR / "circor_splits.csv", index=False))

    n_pat = man["patient_id"].nunique()
    n_pres = (man.drop_duplicates("patient_id")["label"] == 1).sum()
    print(f"[circor manifest] patients={n_pat} (present={n_pres} absent={n_pat - n_pres}) "
          f"recordings={len(man)}  train_pat={len(pat_train)} test_pat={len(pat_test)}",
          flush=True)
    return man


def _load_band(filepath, split):
    """Load, resample, bandpass and peak-normalise one recording."""
    y = load_resampled(filepath, target_sr=SR)
    yb = peak_normalize(bandpass_sos(y, BP_LOW, BP_HIGH, fs=SR, order=BP_ORDER))
    hop_s = 1.5 if split == "train" else 3.0
    return list(segment_fixed(yb, win_s=3.0, hop_s=hop_s, fs=SR))


def _classical_one(task):
    """Process-pool worker: feature vectors for every window of one recording."""
    fp, split, label, pid, rec = task
    try:
        windows = _load_band(fp, split)
    except Exception:
        return []
    out = []
    for w in windows:
        vec = window_feature_vector(w, sr=SR, include_spectral=True)
        out.append((vec.astype("float32"), label, pid, split, rec))
    return out


def _extract_spec(man):
    """Single-process log-mel extraction (cheap: one mel per window)."""
    mel = make_mel(BP_LOW, BP_HIGH, sr=SR)
    X, labels, pids, splits, recs = [], [], [], [], []
    for i, row in enumerate(man.itertuples(index=False)):
        try:
            windows = _load_band(row.filepath, str(row.split))
        except Exception as e:
            print(f"  [skip spec] {row.filepath}: {e}", flush=True)
            continue
        for w in windows:
            X.append(window_to_logmel(w, mel))
            labels.append(int(row.label))
            pids.append(str(row.patient_id))
            splits.append(str(row.split))
            recs.append(str(row.recording_id))
        if (i + 1) % 500 == 0:
            print(f"  [spec] {i + 1}/{len(man)} recordings, windows={len(labels)}", flush=True)
    X = np.asarray(X, dtype="float32")
    assert np.all(np.isfinite(X)), "NaN/Inf in CirCor spectrograms"
    return {
        "X": X,
        "labels": np.asarray(labels, dtype=int),
        "patient_id": np.asarray(pids, dtype=object),
        "split": np.asarray(splits, dtype=object),
        "recording_id": np.asarray(recs, dtype=object),
    }


def _extract_classical(man, workers):
    """Parallel classical-feature extraction across recordings."""
    tasks = [(r.filepath, str(r.split), int(r.label), str(r.patient_id), str(r.recording_id))
             for r in man.itertuples(index=False)]
    feats, labels, pids, splits, recs = [], [], [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_classical_one, tasks, chunksize=8):
            for vec, label, pid, split, rec in res:
                feats.append(vec)
                labels.append(label)
                pids.append(pid)
                splits.append(split)
                recs.append(rec)
            done += 1
            if done % 500 == 0:
                print(f"  [classical] {done}/{len(tasks)} recordings, windows={len(labels)}", flush=True)
    X_B = np.asarray(feats, dtype="float32")
    X_A = X_B[:, :240]
    assert np.all(np.isfinite(X_B)), "NaN/Inf in CirCor classical features"
    return {
        "X_A": X_A, "X_B": X_B,
        "labels": np.asarray(labels, dtype=int),
        "patient_id": np.asarray(pids, dtype=object),
        "split": np.asarray(splits, dtype=object),
        "recording_id": np.asarray(recs, dtype=object),
        "feature_names_A": FEATURE_NAMES_A, "feature_names_B": FEATURE_NAMES_B,
    }


def main():
    ap = argparse.ArgumentParser(description="Build CirCor murmur caches.")
    ap.add_argument("--mode", choices=["spec", "classical", "both"], default="both")
    ap.add_argument("--workers", type=int, default=24,
                    help="Process-pool size for the classical pass.")
    args = ap.parse_args()

    man = _build_manifest()
    assert_no_patient_leakage(
        man.loc[man.split == "train", "patient_id"],
        man.loc[man.split == "test", "patient_id"],
    )
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("spec", "both"):
        spec = _extract_spec(man)
        p = FEATURES_DIR / "circor_spectrograms.npy"
        np.save(p, spec, allow_pickle=True)
        sp = spec["split"]
        print(f"[spec OK] X={spec['X'].shape} train={(sp=='train').sum()} test={(sp=='test').sum()} "
              f"recordings={len(set(spec['recording_id']))} patients={len(set(spec['patient_id']))} "
              f"labels={np.bincount(spec['labels'])} -> {p} ({p.stat().st_size/1e6:.1f} MB)", flush=True)

    if args.mode in ("classical", "both"):
        clf = _extract_classical(man, args.workers)
        p = FEATURES_DIR / "circor_classical.npy"
        np.save(p, clf, allow_pickle=True)
        sp = clf["split"]
        print(f"[classical OK] X_B={clf['X_B'].shape} train={(sp=='train').sum()} test={(sp=='test').sum()} "
              f"-> {p} ({p.stat().st_size/1e6:.1f} MB)", flush=True)

    print("BUILD_CIRCOR_DONE", flush=True)


if __name__ == "__main__":
    main()
