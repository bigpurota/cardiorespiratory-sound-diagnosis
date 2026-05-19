"""
Log-mel spectrogram transform.

Turns a fixed 12000-sample (3.0 s @ 4000 Hz) window/cycle into a ``(64, 128)``
float32 log-mel "image" consumed by the CNN and EfficientNet-B0 models, using a
torchaudio ``MelSpectrogram -> AmplitudeToDB`` stack.

``hop=94`` on a 12000-sample window yields exactly ``1 + 12000 // 94 = 128`` time
frames. ``n_fft=512`` (not 256) avoids torchaudio's empty-mel-filterbank warning on
the narrow heart 20–400 Hz band; it is used for both modalities. fmin/fmax come from
the per-modality bandpass: heart ``make_mel(20, 400)``, lung ``make_mel(200, 1800)``.
"""
from src import config  # noqa: F401 — import first to seed RNGs deterministically

import numpy as np
import torch
import torchaudio.transforms as T

__all__ = ["make_mel", "window_to_logmel"]

# Fixed contract: 12000-sample window -> (64 mel bins, 128 frames).
N_MELS = 64
N_FRAMES = 128
WINDOW_SAMPLES = 12000  # 3.0 s @ 4000 Hz


def make_mel(fmin, fmax, sr=4000, n_fft=512, hop=94, n_mels=64):
    """Build the log-mel transform stack for a band ``[fmin, fmax]`` (build once per modality).

    Returns a ``torch.nn.Sequential`` of ``MelSpectrogram -> AmplitudeToDB``.

    Parameters
    ----------
    fmin, fmax : float
        Mel band edges (Hz); heart 20/400, lung 200/1800.
    sr : int
        Sample rate (Hz); 4000 for both modalities.
    n_fft, hop, n_mels : int
        Keep n_fft=512, hop=94, n_mels=64 for the (64, 128) output contract.
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

    The caller must pad/trim lung cycles to exactly 12000 samples before this call.
    """
    x = torch.as_tensor(window_12000, dtype=torch.float32)
    spec = mel(x)  # (64, 128) dB
    assert spec.shape == (64, 128), f"log-mel shape drift: expected (64, 128) got {tuple(spec.shape)}"
    return spec.numpy().astype("float32")
