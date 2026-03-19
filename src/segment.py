"""
Fixed-window segmenter.

Pure function: slides a fixed-length window over a 1-D signal, drops the ragged
tail and near-silent windows, and returns in-memory ndarrays (nothing is written
to disk). Every emitted segment has the same shape ``(int(win_s*fs),)`` — 12000
samples for the heart 3.0 s @ 4000 Hz default.
"""
from src import config

import numpy as np

__all__ = ["segment_fixed"]


def segment_fixed(y, win_s=3.0, hop_s=1.5, fs=4000, max_silence_fraction=0.8, eps=1e-4):
    """Slice ``y`` into fixed ``win_s``-second windows at ``hop_s`` hop.

    Parameters
    ----------
    y : array-like
        1-D mono signal sampled at ``fs`` Hz.
    win_s, hop_s : float
        Window length and hop in seconds. With the defaults (3.0 s window,
        1.5 s hop @ 4000 Hz) each window is exactly 12000 samples with 50% overlap.
    fs : int
        Sample rate (Hz).
    max_silence_fraction : float
        Drop a window when more than this fraction of its samples are near-zero
        (``|sample| < eps``) — removes effectively-silent windows.
    eps : float
        Near-zero threshold for the silence guard.

    Returns
    -------
    list[np.ndarray]
        Windows, each of shape ``(int(win_s*fs),)``. The ragged tail (a final
        partial window) is dropped. May be empty if every window is silent.
    """
    y = np.asarray(y, dtype="float32")
    win, hop = int(win_s * fs), int(hop_s * fs)

    segs = []
    for start in range(0, max(1, len(y) - win + 1), hop):
        seg = y[start:start + win]
        if len(seg) < win:
            continue
        if np.mean(np.abs(seg) < eps) > max_silence_fraction:
            continue
        segs.append(seg)

    if segs:
        assert all(s.shape == segs[0].shape == (win,) for s in segs), (
            f"SHAPE DRIFT: expected every segment ({win},), got "
            f"{[s.shape for s in segs if s.shape != (win,)][:3]}"
        )
    return segs
