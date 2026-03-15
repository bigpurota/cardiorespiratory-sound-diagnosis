"""
tests/test_config_fixed.py — DATA-04 config-fix assertions (Wave 0, RED).

Encodes the decisions D-01 / D-02 / D-08 from 02-RESEARCH.md:
  - D-01  config.py sampling rates unify to 4000 Hz for BOTH modalities
          (currently SR_HEART=2000, SR_LUNG=8000 — these tests fail until fixed).
  - D-02  params/heart.yaml and params/lung.yaml `sample_rate` become 4000
          (currently 2000 / 8000 respectively).
  - D-08  heart `label_map` de-inversion: PhysioNet/CinC convention is
          -1 -> normal, 1 -> abnormal. params/heart.yaml is currently INVERTED
          (1: normal, -1: abnormal) and must be corrected.

These assertions are INTENTIONALLY RED in Wave 0: no implementation/config edits
exist yet. Collection MUST succeed with zero errors — imports of `config` and
`load_params` are done INSIDE the test bodies so a transient import problem skips
the test rather than erroring at collection time. This mirrors the
sys.path-insert + conftest pattern in tests/test_data_integrity.py.
"""
import pathlib
import sys

import pytest

# Make the project root importable so `import config` and `from src.config_loader
# import load_params` resolve when running pytest directly from the project root.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity


def _import_config():
    """Import the top-level config module inside a test body.

    Skips (does not error) if the import fails for a transient reason so that
    collection always succeeds.
    """
    try:
        import config  # noqa: WPS433  (intentional in-body import)
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"config.py not importable yet: {exc}")
    return config


def _load_params(modality):
    """Import load_params and call it for `modality`, skipping on import failure."""
    try:
        from src.config_loader import load_params
    except Exception as exc:  # pragma: no cover - defensive
        pytest.skip(f"src.config_loader not importable yet: {exc}")
    return load_params(modality)


# ---------------------------------------------------------------------------
# D-01 — config.py sampling rates unified to 4000 Hz
# ---------------------------------------------------------------------------

def test_config_sr_4000():
    """config.SR_HEART and config.SR_LUNG must both equal 4000 Hz (D-01)."""
    config = _import_config()
    assert config.SR_HEART == 4000, (
        f"config.SR_HEART must be 4000 (D-01), got {getattr(config, 'SR_HEART', None)}"
    )
    assert config.SR_LUNG == 4000, (
        f"config.SR_LUNG must be 4000 (D-01), got {getattr(config, 'SR_LUNG', None)}"
    )


# ---------------------------------------------------------------------------
# D-02 — params YAML sample_rate fixed to 4000 Hz for both modalities
# ---------------------------------------------------------------------------

def test_heart_yaml_sample_rate():
    """params/heart.yaml `sample_rate` must be 4000 (was 2000) — D-02."""
    params = _load_params("heart")
    assert params["sample_rate"] == 4000, (
        f"params/heart.yaml sample_rate must be 4000 (D-02), got {params['sample_rate']}"
    )


def test_lung_yaml_sample_rate():
    """params/lung.yaml `sample_rate` must be 4000 (was 8000) — D-02."""
    params = _load_params("lung")
    assert params["sample_rate"] == 4000, (
        f"params/lung.yaml sample_rate must be 4000 (D-02), got {params['sample_rate']}"
    )


# ---------------------------------------------------------------------------
# D-08 — heart label_map de-inversion: -1 -> normal, 1 -> abnormal
# ---------------------------------------------------------------------------

def test_heart_label_map_deinverted():
    """heart label_map must resolve -1 -> normal and 1 -> abnormal (D-08).

    PhysioNet/CinC 2016 convention (verified against REFERENCE.csv and the .hea
    `# Normal`/`# Abnormal` comments): -1 is normal, 1 is abnormal. The YAML keys
    may be parsed as ints or strings, so accept both spellings.
    """
    params = _load_params("heart")
    label_map = params["label_map"]

    def _lookup(key):
        for candidate in (key, str(key)):
            if candidate in label_map:
                return str(label_map[candidate]).lower()
        raise AssertionError(f"label_map missing key {key!r}: {label_map!r}")

    assert _lookup(-1) == "normal", (
        f"label_map[-1] must be 'normal' (D-08 de-inversion), got {_lookup(-1)!r}"
    )
    assert _lookup(1) == "abnormal", (
        f"label_map[1] must be 'abnormal' (D-08 de-inversion), got {_lookup(1)!r}"
    )
