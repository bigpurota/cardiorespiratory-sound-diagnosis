"""Pure audio preprocessing primitives."""
from src import config

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, sosfilt

__all__ = ["resample", "load_resampled", "bandpass_sos", "peak_normalize"]


def resample(y, orig_sr, target_sr=4000):
    """Resample a 1-D signal from ``orig_sr`` to ``target_sr``"""
    y = np.asarray(y, dtype="float32")
    if orig_sr == target_sr:
        return y
    out = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    return out.astype("float32")


def load_resampled(path, target_sr=4000):
    """Load an audio file at its native rate and resample to"""
    y, sr = librosa.load(path, sr=None)
    return resample(y, orig_sr=sr, target_sr=target_sr)


def bandpass_sos(y, fmin, fmax, fs=4000, order=4):
    """Zero-phase Butterworth bandpass via second-order sections"""
    y = np.asarray(y, dtype="float32")
    sos = butter(order, [fmin, fmax], btype="band", fs=fs, output="sos")

    padlen = 3 * (2 * sos.shape[0])
    if y.shape[0] <= padlen:
        out = sosfilt(sos, y)
    else:
        out = sosfiltfilt(sos, y)
    return np.asarray(out, dtype="float32")


def peak_normalize(y, eps=1e-9):
    """Per-clip peak normalization to [-1, 1] (float32)."""
    y = np.asarray(y, dtype="float32")
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m < eps:
        return y
    return (y / m).astype("float32")
