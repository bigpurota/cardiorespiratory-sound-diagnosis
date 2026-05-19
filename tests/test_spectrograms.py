"""Tests for the log-mel spectrogram helpers in src/spectrograms.py.

make_mel builds a torchaudio MelSpectrogram → AmplitudeToDB stack and
window_to_logmel turns a 12000-sample window into a (64, 128) float32 dB tensor
(hop=94 → 1 + 12000//94 = 128 frames). n_fft=512 also avoids the empty
mel-filterbank warning that the heart 20–400 Hz band raises at a smaller FFT.
Imports happen inside the test bodies and skip when the module is unavailable.
"""
import importlib
import pathlib
import sys
import warnings

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

WINDOW_SAMPLES = 12000  # 3.0 s @ 4000 Hz — the fixed window (matches features)


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive (module absent)
        pytest.skip(f"{module_name} not implemented yet: {exc}")


# window_to_logmel returns the expected (64,128) float32 dB image

def test_shape_dtype():
    """``window_to_logmel(window, make_mel(20,400))`` returns shape (64,128) dtype float32.

    A 12000-sample window through the heart-band mel (n_fft=512, hop=94, n_mels=64)
    yields exactly 64 mel bins × 128 frames as a float32 array.
    """
    spectrograms = _import("src.spectrograms")
    for fn in ("make_mel", "window_to_logmel"):
        if not hasattr(spectrograms, fn):
            pytest.skip(f"src.spectrograms.{fn} not implemented yet")

    import numpy as np

    rng = np.random.default_rng(42)
    window = rng.standard_normal(WINDOW_SAMPLES).astype("float32")

    mel = spectrograms.make_mel(20, 400)
    spec = spectrograms.window_to_logmel(window, mel)

    spec = np.asarray(spec)
    assert spec.shape == (64, 128), f"expected (64,128) log-mel, got {spec.shape}"
    assert spec.dtype == np.float32, f"expected float32, got {spec.dtype}"


# n_fft=512 emits no empty-mel-filterbank warning

def test_no_filterbank_warning():
    """Building the heart 20–400 Hz mel (n_fft=512) emits NO empty-filterbank warning.

    The heart band (20–400 Hz) with a small n_fft=256 triggers torchaudio's
    "mel filterbank has all zero values" UserWarning (some mel bins span no FFT bin).
    make_mel uses n_fft=512 so no such warning is emitted. We construct the mel and
    run one window through it inside a warnings-recording context.
    """
    spectrograms = _import("src.spectrograms")
    for fn in ("make_mel", "window_to_logmel"):
        if not hasattr(spectrograms, fn):
            pytest.skip(f"src.spectrograms.{fn} not implemented yet")

    import numpy as np

    rng = np.random.default_rng(42)
    window = rng.standard_normal(WINDOW_SAMPLES).astype("float32")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mel = spectrograms.make_mel(20, 400)        # heart band → n_fft must be 512
        spectrograms.window_to_logmel(window, mel)

    offenders = [
        str(w.message) for w in caught
        if "mel filterbank" in str(w.message).lower()
        or "all zero" in str(w.message).lower()
    ]
    assert not offenders, (
        "empty mel-filterbank warning emitted for the heart 20-400 Hz band — "
        f"make_mel must use n_fft=512. Got: {offenders}"
    )
