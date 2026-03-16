"""
src/preprocess.py — pure audio preprocessing functions (Phase 2, D-05).

Implements the resample → zero-phase Butterworth bandpass → peak-normalize
primitives from 02-RESEARCH.md Code Examples §6. These are *pure* ndarray→ndarray
functions: Phase 2 validates them with unit tests but does NOT materialise any
preprocessed audio or `.npy` caches to disk (D-05 — that happens in Phase 3).

`import config` runs first (its module-level side effect seeds random/numpy/torch
at SEED=42) so every consumer inherits deterministic RNG state.
"""
import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, sosfilt

__all__ = ["resample", "load_resampled", "bandpass_sos", "peak_normalize"]


def resample(y, orig_sr, target_sr=4000):
    """Resample a 1-D signal from ``orig_sr`` to ``target_sr`` (float32).

    Uses ``librosa.resample`` (``soxr_hq``). A 2000 Hz → 4000 Hz resample yields
    exactly 2× the input sample count (verified on a real heart recording).
    Returns the input unchanged (as float32) when the rates already match.
    """
    y = np.asarray(y, dtype="float32")
    if orig_sr == target_sr:
        return y
    out = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    return out.astype("float32")


def load_resampled(path, target_sr=4000):
    """Load an audio file at its native rate and resample to ``target_sr``.

    Reads the native sampling rate (``sr=None``) then delegates to :func:`resample`
    so on-disk loads and in-memory resampling share one code path.
    """
    y, sr = librosa.load(path, sr=None)
    return resample(y, orig_sr=sr, target_sr=target_sr)


def bandpass_sos(y, fmin, fmax, fs=4000, order=4):
    """Zero-phase Butterworth bandpass via second-order sections (float32).

    Builds an order-``order`` SOS bandpass for ``[fmin, fmax]`` at sample rate
    ``fs`` and applies it with ``sosfiltfilt`` (zero phase, preserves waveform
    timing). For inputs shorter than ``sosfiltfilt``'s internal pad length the
    filter would raise ``ValueError``; we guard by falling back to the causal
    one-pass ``sosfilt`` so very short segments (e.g. n=30) stay finite and
    same-length (Pitfall 3 — sosfiltfilt padlen on short segments).
    """
    y = np.asarray(y, dtype="float32")
    sos = butter(order, [fmin, fmax], btype="band", fs=fs, output="sos")

    # sosfiltfilt requires len(y) > padlen; padlen scales with the number of
    # sections. Fall back to causal sosfilt for inputs at/below that threshold.
    padlen = 3 * (2 * sos.shape[0])  # scipy default ntaps for filtfilt padding
    if y.shape[0] <= padlen:
        out = sosfilt(sos, y)
    else:
        out = sosfiltfilt(sos, y)
    return np.asarray(out, dtype="float32")


def peak_normalize(y, eps=1e-9):
    """Per-clip peak normalization to [-1, 1] (float32).

    Divides by the maximum absolute value. Per-clip ONLY — no global/StandardScaler
    fitting here (that is a Phase-3 train-fold concern). Near-silent clips (peak
    below ``eps``) are returned unchanged to avoid division blow-up.
    """
    y = np.asarray(y, dtype="float32")
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m < eps:
        return y
    return (y / m).astype("float32")
