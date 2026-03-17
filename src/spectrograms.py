"""
src/spectrograms.py — log-mel spectrogram transform (Phase 4, MODL-02).

The single genuinely new ML representation in Phase 4: turn a fixed 12000-sample
(3.0 s @ 4000 Hz) window/cycle into a ``(64, 128)`` float32 log-mel "image" that the
small CNN and the EfficientNet-B0 transfer model consume. Mirrors the role of
``src/features.py`` (a pure window → fixed-length representation) but swaps the classical
MFCC vector for a torchaudio ``MelSpectrogram → AmplitudeToDB`` stack.

VERIFIED recipe (04-RESEARCH.md §Code Examples 1, live in the project .venv):
  - ``make_mel(fmin, fmax, sr=4000, n_fft=512, hop=94, n_mels=64)`` →
    ``MelSpectrogram(power=2.0, center=True) → AmplitudeToDB(stype="power", top_db=80.0)``.
  - ``window_to_logmel(window_12000, mel)`` → ``(64, 128)`` float32 dB array.
  - ``hop=94`` on a 12000-sample window yields EXACTLY ``1 + 12000 // 94 = 128`` time
    frames (``center=True``); the inline ``assert spec.shape == (64, 128)`` never trips.
  - ``n_fft=512`` (257 FFT bins) — NOT 256 — avoids torchaudio's "mel filterbank has all
    zero values" UserWarning that the narrow heart 20–400 Hz band raises at the smaller
    FFT size (Pitfall 1). Used for BOTH modalities for uniformity (lung 200–1800 Hz is
    clean at any n_fft).

fmin/fmax align to ``params/{modality}.yaml`` bandpass: heart ``make_mel(20, 400)``,
lung ``make_mel(200, 1800)``. No librosa mel code here — torchaudio is the chosen,
GPU-native path (librosa is the documented fallback only, 04-RESEARCH §Alternatives).

``import config`` runs first for the SEED=42 determinism side effect.
"""
import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import numpy as np
import torch
import torchaudio.transforms as T

__all__ = ["make_mel", "window_to_logmel"]

# Canonical fixed contract (Pitfall 2): 12000-sample window → (64 mel bins, 128 frames).
N_MELS = 64
N_FRAMES = 128
WINDOW_SAMPLES = 12000  # 3.0 s @ 4000 Hz


def make_mel(fmin, fmax, sr=4000, n_fft=512, hop=94, n_mels=64):
    """Build the log-mel transform stack for a band ``[fmin, fmax]`` (build ONCE per modality).

    Returns a ``torch.nn.Sequential`` of ``MelSpectrogram(power=2.0, center=True) →
    AmplitudeToDB(stype="power", top_db=80.0)``. ``n_fft=512`` (not 256) is mandatory so
    the narrow heart 20–400 Hz band does not raise the empty-mel-filterbank warning
    (Pitfall 1); ``hop=94`` gives exactly 128 frames on a 12000-sample window (Pitfall 2).

    Parameters
    ----------
    fmin, fmax : float
        Mel band edges (Hz) — derive from ``params/{modality}.yaml`` bandpass
        (heart 20/400, lung 200/1800).
    sr : int
        Sample rate (Hz); 4000 for both modalities (D-01).
    n_fft, hop, n_mels : int
        VERIFIED recipe — keep n_fft=512, hop=94, n_mels=64 for the (64,128) contract.
    """
    return torch.nn.Sequential(
        T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
            center=True,
        ),
        T.AmplitudeToDB(stype="power", top_db=80.0),
    )


def window_to_logmel(window_12000, mel):
    """Turn one 12000-sample window/cycle into a ``(64, 128)`` float32 log-mel dB array.

    Runs ``window`` through the prebuilt ``mel`` stack, enforces the fixed shape via an
    inline ``assert spec.shape == (64, 128)`` (V5 shape guard / T-04-02), and returns a
    contiguous float32 numpy array (cache-ready, GPU-copy-free). The caller is responsible
    for padding/trimming lung cycles to exactly 12000 samples before this call (same rule
    as ``src/features.lung_cycle_vector``).
    """
    x = torch.as_tensor(window_12000, dtype=torch.float32)
    spec = mel(x)  # (64, 128) dB
    assert spec.shape == (64, 128), f"log-mel shape drift: expected (64, 128) got {tuple(spec.shape)}"
    return spec.numpy().astype("float32")
