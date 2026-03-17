"""
src/features.py — classical feature extraction (Phase 3, DATA-05).

Pure, leakage-safe feature extraction for the classical comparative study:

  - ``window_feature_vector(w, sr=4000, include_spectral=False)`` — one fixed-length
    vector per signal window: MFCC(n_mfcc=40) + Δ + ΔΔ summarised as mean+std over
    frames → 240-d (Set A); ``include_spectral=True`` appends 5 spectral statistics
    (centroid, rolloff, bandwidth, zero-crossing-rate, RMS) each (mean, std) → 250-d
    (Set B). Returns float32. (03-RESEARCH.md §Pattern 1.)
  - ``lung_cycle_vector(yb, start_s, end_s, sr=4000, pad_s=3.0, ...)`` — slices a
    respiratory cycle, then **pads/trims to 3.0 s (12000 samples) BEFORE MFCC** so
    ``librosa.feature.delta`` (default width=9) never raises ``ParameterError`` on a
    short cycle (a raw 0.2-s cycle yields only 2 MFCC frames and would crash delta —
    03-RESEARCH.md §Pattern 2 / §Pitfall 2). Delegates to ``window_feature_vector``.
  - ``extract_features(modality, df, splits_df, params, include_spectral_both=True)`` —
    orchestration: runs the Phase-2 preprocess chain
    (load_resampled → bandpass_sos → peak_normalize) per recording and emits aligned
    arrays (X_A 240-d, X_B 250-d, labels, group/patient_id, split, recording_id) plus
    the feature-name schema. No global StandardScaler is fit here (D-05); no raw-audio
    cache is written (features only).

Label encodings (verified against on-disk data, 03-RESEARCH.md §Pattern 2):
  - Heart manifest label is float {-1.0 normal, 1.0 abnormal} → {0 normal, 1 abnormal};
    recording_id == patient_id (one heart recording per patient).
  - Lung cycle label strings are {crackle, wheeze, both, normal} (NOTE: ``crackle`` is
    SINGULAR on disk; params/lung.yaml's ``crackles`` plural is dormant) →
    {crackle:0, wheeze:1, both:2, normal:3} so the pooled-abnormal mask is ``label != 3``.

``import config`` runs first for the SEED=42 determinism side effect.
"""
import os
import pathlib

import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import numpy as np
import librosa

from src.preprocess import load_resampled, bandpass_sos, peak_normalize
from src.segment import segment_fixed

__all__ = [
    "window_feature_vector",
    "lung_cycle_vector",
    "extract_features",
    "feature_names",
    "FEATURE_NAMES_A",
    "FEATURE_NAMES_B",
    "HEART_LABEL_MAP",
    "LUNG_LABEL_MAP",
]

# Heart manifest float convention {-1.0 normal, 1.0 abnormal} → {0 normal, 1 abnormal}.
HEART_LABEL_MAP = {-1.0: 0, 1.0: 1, -1: 0, 1: 1}

# Lung cycle strings (on disk) → integer codes; ``normal`` is the highest index so the
# pooled-abnormal mask used by icbhi_score is simply ``label != 3``.
LUNG_LABEL_MAP = {"crackle": 0, "wheeze": 1, "both": 2, "normal": 3}

# Spectral statistic names appended for Set B (each contributes a mean and a std).
_SPECTRAL_NAMES = ["centroid", "rolloff", "bandwidth", "zcr", "rms"]


# ---------------------------------------------------------------------------
# Feature-name schema (mirrors the concatenation order in window_feature_vector)
# ---------------------------------------------------------------------------
def feature_names(include_spectral=False, n_mfcc=40):
    """Return the ordered feature-name list matching ``window_feature_vector``.

    Order: mean over frames of [mfcc, Δ, ΔΔ] (3·n_mfcc), then std over frames of
    [mfcc, Δ, ΔΔ] (3·n_mfcc) → 240 for Set A; Set B appends, per spectral stat,
    ``{name}_mean`` then ``{name}_std`` → +10 = 250.
    """
    names = []
    for stat in ("mean", "std"):
        for block in ("mfcc", "d1", "d2"):
            names += [f"{block}_{i}_{stat}" for i in range(n_mfcc)]
    if include_spectral:
        for sp in _SPECTRAL_NAMES:
            names += [f"{sp}_mean", f"{sp}_std"]
    return names


FEATURE_NAMES_A = feature_names(include_spectral=False)
FEATURE_NAMES_B = feature_names(include_spectral=True)


# ---------------------------------------------------------------------------
# §Pattern 1 — heart/window feature vector (Set A 240-d / Set B 250-d)
# ---------------------------------------------------------------------------
def window_feature_vector(w, sr=4000, include_spectral=False):
    """Extract a fixed 240-d (Set A) / 250-d (Set B) float32 feature vector.

    MFCC(n_mfcc=40) + Δ + ΔΔ summarised as mean then std over frames → 240-d.
    When ``include_spectral`` is true, append (mean, std) of spectral centroid,
    rolloff, bandwidth, zero-crossing-rate and RMS → 250-d. (03-RESEARCH.md §Pattern 1.)
    """
    w = np.asarray(w, dtype="float32")
    mfcc = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=40)   # (40, frames); librosa defaults n_fft=2048, hop=512
    d1 = librosa.feature.delta(mfcc)                     # default width=9 — OK: 3-s window → 24 frames
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = [f.mean(axis=1) for f in (mfcc, d1, d2)] + [f.std(axis=1) for f in (mfcc, d1, d2)]
    vec = np.concatenate(feats)                          # 6 * 40 = 240 (Set A)
    if include_spectral:
        cent = librosa.feature.spectral_centroid(y=w, sr=sr)
        roll = librosa.feature.spectral_rolloff(y=w, sr=sr)
        bw = librosa.feature.spectral_bandwidth(y=w, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(w)
        rms = librosa.feature.rms(y=w)
        spec = np.concatenate([[f.mean(), f.std()] for f in (cent, roll, bw, zcr, rms)])  # 10 (Set B add)
        vec = np.concatenate([vec, spec])                # 250 (Set B)
    return vec.astype("float32")


# ---------------------------------------------------------------------------
# §Pattern 2 — lung cycle feature vector — PAD BEFORE EXTRACT (mandatory)
# ---------------------------------------------------------------------------
def lung_cycle_vector(yb, start_s, end_s, sr=4000, pad_s=3.0, include_spectral=False):
    """Slice a respiratory cycle and pad/trim to ``pad_s`` s BEFORE MFCC, then extract.

    Padding before MFCC is mandatory: a short (e.g. 0.2-s) cycle raw yields only 2
    MFCC frames and ``librosa.feature.delta`` (width=9) raises ``ParameterError``.
    Padding to 12000 samples first yields 24 frames so delta works. (§Pattern 2 / §Pitfall 2.)
    """
    yb = np.asarray(yb, dtype="float32")
    s, e = int(start_s * sr), int(end_s * sr)
    cyc = yb[s:e]
    target = int(pad_s * sr)                              # 12000 @ 4000 Hz
    if len(cyc) < target:
        cyc = np.pad(cyc, (0, target - len(cyc)))         # zero-pad short cycles
    else:
        cyc = cyc[:target]                                # trim long cycles
    return window_feature_vector(cyc, sr=sr, include_spectral=include_spectral)


# ---------------------------------------------------------------------------
# Orchestration — preprocess chain + per-window/per-cycle vectors → aligned arrays
# ---------------------------------------------------------------------------
def _assert_vector_ok(vec, expect_dim):
    """Data-integrity guard (T-03-V5a): fixed shape + finite (no NaN/Inf)."""
    assert vec.shape == (expect_dim,), f"feature shape drift: expected ({expect_dim},) got {vec.shape}"
    assert np.all(np.isfinite(vec)), "feature vector contains NaN/Inf"


def extract_features(modality, df, splits_df, params, include_spectral_both=True):
    """Run preprocess+features for ``modality`` → aligned feature arrays + metadata.

    Parameters
    ----------
    modality : {"heart", "lung"}
        Heart reads recording rows (one vector per 3-s window; TRAIN windows use a
        1.5-s hop / 50% overlap, TEST windows a 3.0-s hop / no overlap). Lung reads
        per-cycle rows (one vector per respiratory cycle, pad-before-delta).
    df : pandas.DataFrame
        Heart: the manifest filtered to ``modality == "heart"`` (filepath, patient_id,
        label). Lung: lung_cycles.csv (filepath, patient_id, start_s, end_s, label).
    splits_df : pandas.DataFrame
        Patient-level split table (patient_id, split) — joined on patient_id.
    params : dict
        Modality params (bandpass_low_hz, bandpass_high_hz, bandpass_order).
    include_spectral_both : bool
        When True, emit BOTH Set A (240-d) and Set B (250-d). Set B is a superset, so
        we compute the 250-d vector once and slice its first 240 columns for Set A.

    Returns
    -------
    dict
        Keys: X_A (N×240 float32), X_B (N×250 float32), labels (N int), patient_id (N),
        split (N), recording_id (N), feature_names_A, feature_names_B.
    """
    import pandas as pd

    fmin = int(params.get("bandpass_low_hz"))
    fmax = int(params.get("bandpass_high_hz"))
    order = int(params.get("bandpass_order", 4))
    sr = int(params.get("sample_rate", 4000))

    split_lookup = {str(r.patient_id): str(r.split) for r in splits_df.itertuples()}

    X_B_rows, labels, pids, splits, rec_ids = [], [], [], [], []

    if modality == "heart":
        for row in df.itertuples():
            pid = str(row.patient_id)
            split = split_lookup.get(pid)
            if split is None:
                continue  # recording not in the (within-A–E) split → skip
            y = load_resampled(row.filepath, target_sr=sr)
            yb = peak_normalize(bandpass_sos(y, fmin, fmax, fs=sr, order=order))
            # TRAIN: 50% overlap (hop_s=1.5); TEST: no overlap (hop_s=3.0) — changes only
            # the majority-vote denominator, not correctness (params/heart.yaml comment).
            hop_s = 1.5 if split == "train" else 3.0
            label = HEART_LABEL_MAP.get(float(row.label))
            if label is None:
                continue  # unsure / out-of-domain label → exclude from training targets
            for w in segment_fixed(yb, win_s=3.0, hop_s=hop_s, fs=sr):
                vec = window_feature_vector(w, sr=sr, include_spectral=True)
                _assert_vector_ok(vec, 250)
                X_B_rows.append(vec)
                labels.append(label)
                pids.append(pid)
                splits.append(split)
                rec_ids.append(pid)  # heart recording_id == patient_id
    elif modality == "lung":
        # Preprocess each recording ONCE, then slice all its cycles (avoid re-loading).
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
                    continue  # unknown label string → skip (data-integrity guard)
                vec = lung_cycle_vector(
                    yb, float(cyc.start_s), float(cyc.end_s), sr=sr, include_spectral=True
                )
                _assert_vector_ok(vec, 250)
                X_B_rows.append(vec)
                labels.append(label)
                pids.append(pid)
                splits.append(split)
                rec_ids.append(rec_id)
    else:
        raise ValueError(f"Unknown modality '{modality}'. Expected 'heart' or 'lung'.")

    X_B = np.asarray(X_B_rows, dtype="float32") if X_B_rows else np.empty((0, 250), dtype="float32")
    X_A = X_B[:, :240]  # Set A is the first 240 columns of Set B (same concatenation order)
    assert np.all(np.isfinite(X_B)), "cached feature matrix contains NaN/Inf"

    return {
        "X_A": X_A,
        "X_B": X_B,
        "labels": np.asarray(labels, dtype=int),
        "patient_id": np.asarray(pids, dtype=object),
        "split": np.asarray(splits, dtype=object),
        "recording_id": np.asarray(rec_ids, dtype=object),
        "feature_names_A": FEATURE_NAMES_A,
        "feature_names_B": FEATURE_NAMES_B,
    }
