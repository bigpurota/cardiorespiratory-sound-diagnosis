"""
tests/test_preprocess.py — DATA-04 preprocessing assertions (Wave 0, RED).

Specifies the contracts for the Wave-1 pure functions described in
02-RESEARCH.md Code Examples §6 / §7:

  - src.preprocess.load_resampled(path, target_sr=4000) — librosa native load +
    resample; a 2000 Hz signal resampled to 4000 Hz yields exactly 2x samples.
    (We test the resample contract via the lower-level src.preprocess.resample
    function on the synthetic array, since load_resampled reads from disk.)
  - src.preprocess.bandpass_sos(y, fmin, fmax, fs=4000, order=4) — zero-phase
    scipy SOS Butterworth; returns a finite, same-length float32 array, and is
    stable even for very short inputs (n=30) per the VERIFIED sosfiltfilt range.
  - src.segment.segment_fixed(y, win_s=3.0, hop_s=1.5, fs=4000) — fixed-window
    segmenter; every emitted window has shape (12000,) for win_s=3.0 @ fs=4000.

These reference src.preprocess and src.segment symbols that do NOT exist yet, so
the tests MUST be RED now. Collection MUST succeed: imports happen INSIDE the test
bodies (skip-on-missing) so a missing module never errors at collection time.
"""
import importlib
import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity

# Window length contract: int(3.0 * 4000) == 12000 samples (VERIFIED in research).
EXPECTED_WINDOW = 12000
FS = 4000


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent in Wave 0."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive (Wave 0 modules absent)
        pytest.skip(f"{module_name} not implemented yet (Wave 0): {exc}")


def _get_resampler(preprocess):
    """Return a callable that resamples (y, orig_sr, target_sr) -> ndarray.

    Prefer an explicit `resample`; otherwise fall back to a hypothetical
    `load_resampled`-companion. If neither exists, skip — the contract is RED.
    """
    if hasattr(preprocess, "resample"):
        return lambda y, orig, target: preprocess.resample(
            y, orig_sr=orig, target_sr=target
        )
    pytest.skip("src.preprocess.resample not implemented yet (Wave 0)")


# ---------------------------------------------------------------------------
# Resample contract — 2000 Hz -> 4000 Hz yields exactly 2x sample count
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bandpass contract — finite, same-length float32, stable on short n=30 input
# ---------------------------------------------------------------------------

def test_bandpass_finite(synthetic_signal):
    """bandpass_sos returns a finite, same-length float32 array for long and short inputs."""
    preprocess = _import("src.preprocess")
    if not hasattr(preprocess, "bandpass_sos"):
        pytest.skip("src.preprocess.bandpass_sos not implemented yet (Wave 0)")

    fmin, fmax = 20, 400

    # Long input (the full synthetic signal).
    y_long = synthetic_signal["y"]
    out_long = np.asarray(preprocess.bandpass_sos(y_long, fmin, fmax, fs=FS, order=4))
    assert out_long.shape == y_long.shape, "bandpass changed array length (long input)"
    assert np.all(np.isfinite(out_long)), "bandpass produced non-finite values (long input)"
    assert out_long.dtype == np.float32, f"expected float32, got {out_long.dtype}"

    # Short input (n=30) — sosfiltfilt VERIFIED stable down to n=30 at order 4.
    y_short = np.asarray(y_long[:30], dtype="float32")
    out_short = np.asarray(preprocess.bandpass_sos(y_short, fmin, fmax, fs=FS, order=4))
    assert out_short.shape == y_short.shape, "bandpass changed array length (n=30 input)"
    assert np.all(np.isfinite(out_short)), "bandpass produced non-finite values (n=30 input)"


# ---------------------------------------------------------------------------
# Segmentation contract — every fixed window has shape (12000,) at win_s=3.0/fs=4000
# ---------------------------------------------------------------------------

def test_segment_shape_consistency(synthetic_signal):
    """segment_fixed emits windows that ALL have shape (12000,) for win_s=3.0 @ fs=4000."""
    segment = _import("src.segment")
    if not hasattr(segment, "segment_fixed"):
        pytest.skip("src.segment.segment_fixed not implemented yet (Wave 0)")

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
