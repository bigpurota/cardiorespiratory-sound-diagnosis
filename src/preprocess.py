"""
Pure audio preprocessing primitives.

resample -> zero-phase Butterworth bandpass -> peak-normalize, all as pure
ndarray -> ndarray functions (nothing is written to disk here).
"""
from src import config  # noqa: F401 — import first to seed RNGs deterministically

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, sosfilt

__all__ = ["resample", "load_resampled", "bandpass_sos", "peak_normalize"]


def resample(y, orig_sr, target_sr=4000):
    """Resample a 1-D signal from ``orig_sr`` to ``target_sr`` (float32).

    Returns the input unchanged (as float32) when the rates already match.
    """
    y = np.asarray(y, dtype="float32")
    if orig_sr == target_sr:
        return y
    out = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    return out.astype("float32")


def load_resampled(path, target_sr=4000):
    """Load an audio file at its native rate and resample to ``target_sr``."""
    y, sr = librosa.load(path, sr=None)
    return resample(y, orig_sr=sr, target_sr=target_sr)


def bandpass_sos(y, fmin, fmax, fs=4000, order=4):
    """Zero-phase Butterworth bandpass via second-order sections (float32).

    Applies ``sosfiltfilt`` (zero phase, preserves waveform timing). Inputs shorter
    than its internal pad length would raise ``ValueError``, so we fall back to the
    causal one-pass ``sosfilt`` to keep very short segments finite and same-length.
    """
    y = np.asarray(y, dtype="float32")
    sos = butter(order, [fmin, fmax], btype="band", fs=fs, output="sos")

    # sosfiltfilt requires len(y) > padlen; fall back to sosfilt at/below it.
    padlen = 3 * (2 * sos.shape[0])  # scipy default filtfilt pad length
    if y.shape[0] <= padlen:
        out = sosfilt(sos, y)
    else:
        out = sosfiltfilt(sos, y)
    return np.asarray(out, dtype="float32")


def peak_normalize(y, eps=1e-9):
    """Per-clip peak normalization to [-1, 1] (float32).

    Divides by the maximum absolute value (per-clip only — no global scaler fitting).
    Near-silent clips (peak below ``eps``) are returned unchanged to avoid blow-up.
    """
    y = np.asarray(y, dtype="float32")
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m < eps:
        return y
    return (y / m).astype("float32")
