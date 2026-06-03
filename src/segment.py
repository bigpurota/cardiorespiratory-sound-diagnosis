"""Fixed-window segmenter."""
from src import config

import numpy as np

__all__ = ["segment_fixed"]


def segment_fixed(y, win_s=3.0, hop_s=1.5, fs=4000, max_silence_fraction=0.8, eps=1e-4):
    """Slice ``y`` into fixed ``win_s``-second windows at"""
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
