"""Log-mel spectrogram transform."""
from src import config

import numpy as np
import torch
import torchaudio.transforms as T

__all__ = ["make_mel", "window_to_logmel"]

N_MELS = 64
N_FRAMES = 128
WINDOW_SAMPLES = 12000


def make_mel(fmin, fmax, sr=4000, n_fft=512, hop=94, n_mels=64):
    """Build the log-mel transform stack for a band ``[fmin,"""
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
    """Turn one 12000-sample window/cycle into a ``(64, 128)``"""
    x = torch.as_tensor(window_12000, dtype=torch.float32)
    spec = mel(x)
    assert spec.shape == (64, 128), f"log-mel shape drift: expected (64, 128) got {tuple(spec.shape)}"
    return spec.numpy().astype("float32")
