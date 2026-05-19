"""Tests for the preprocessing and segmentation helpers.

Covers src.preprocess.resample (2000 Hz -> 4000 Hz doubles the sample count),
src.preprocess.bandpass_sos (finite, same-length float32 output, stable on short
inputs), and src.segment.segment_fixed (every window is 12000 samples at
win_s=3.0, fs=4000). Imports are done inside the test bodies and skip when a
module is unavailable.
"""
import importlib
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Window length: int(3.0 * 4000) == 12000 samples.
EXPECTED_WINDOW = 12000
FS = 4000


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive (module absent)
        pytest.skip(f"{module_name} not implemented yet: {exc}")


def _get_resampler(preprocess):
    """Return a callable that resamples (y, orig_sr, target_sr) -> ndarray.

    Prefer an explicit `resample`; skip if it is unavailable.
    """
    if hasattr(preprocess, "resample"):
        return lambda y, orig, target: preprocess.resample(
            y, orig_sr=orig, target_sr=target
        )
    pytest.skip("src.preprocess.resample not implemented yet")


# Resample — 2000 Hz -> 4000 Hz yields exactly 2x sample count

def test_resample_shape(synthetic_signal):
    """Resampling the 2000 Hz synthetic array to 4000 Hz gives exactly 2x samples."""
    preprocess = _import("src.preprocess")
    resample = _get_resampler(preprocess)

    y = synthetic_signal["y"]
    orig_sr = synthetic_signal["sr"]            # 2000
    target_sr = 4000

    out = np.asarray(resample(y, orig_sr, target_sr))
    assert out.shape[0] == 2 * y.shape[0], (
        f"2000->4000 resample must double length: expected {2 * y.shape[0]}, "
        f"got {out.shape[0]}"
    )


# Bandpass — finite, same-length float32, stable on short n=30 input

def test_bandpass_finite(synthetic_signal):
    """bandpass_sos returns a finite, same-length float32 array for long and short inputs."""
    preprocess = _import("src.preprocess")
    if not hasattr(preprocess, "bandpass_sos"):
        pytest.skip("src.preprocess.bandpass_sos not implemented yet")

    fmin, fmax = 20, 400

    # Long input (the full synthetic signal).
    y_long = synthetic_signal["y"]
    out_long = np.asarray(preprocess.bandpass_sos(y_long, fmin, fmax, fs=FS, order=4))
    assert out_long.shape == y_long.shape, "bandpass changed array length (long input)"
    assert np.all(np.isfinite(out_long)), "bandpass produced non-finite values (long input)"
    assert out_long.dtype == np.float32, f"expected float32, got {out_long.dtype}"

    # Short input (n=30) — sosfiltfilt stays stable down to n=30 at order 4.
    y_short = np.asarray(y_long[:30], dtype="float32")
    out_short = np.asarray(preprocess.bandpass_sos(y_short, fmin, fmax, fs=FS, order=4))
    assert out_short.shape == y_short.shape, "bandpass changed array length (n=30 input)"
    assert np.all(np.isfinite(out_short)), "bandpass produced non-finite values (n=30 input)"


# Segmentation — every fixed window has shape (12000,) at win_s=3.0/fs=4000

def test_segment_shape_consistency(synthetic_signal):
    """segment_fixed emits windows that ALL have shape (12000,) for win_s=3.0 @ fs=4000."""
    segment = _import("src.segment")
    if not hasattr(segment, "segment_fixed"):
        pytest.skip("src.segment.segment_fixed not implemented yet")

    # Build a >=2-window signal at fs=4000 so segmentation has something to emit.
    rng = np.random.default_rng(42)
    n = int(7.0 * FS)                            # 28000 samples -> multiple 3s windows
    y = (np.sin(2 * np.pi * 100 * np.arange(n) / FS) + 0.05 * rng.standard_normal(n)).astype("float32")

    segs = segment.segment_fixed(y, win_s=3.0, hop_s=1.5, fs=FS)
    assert len(segs) > 0, "segment_fixed returned no windows for a 7-second signal"
    for s in segs:
        s = np.asarray(s)
        assert s.shape == (EXPECTED_WINDOW,), (
            f"every window must be ({EXPECTED_WINDOW},); got {s.shape}"
        )
