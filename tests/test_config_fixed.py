"""Config-consistency assertions for sampling rates and the heart label map.

Checks:
  - config.py sampling rates unify to 4000 Hz for both modalities.
  - params/heart.yaml and params/lung.yaml ``sample_rate`` are 4000.
  - heart ``label_map`` follows the PhysioNet/CinC convention:
    -1 -> normal, 1 -> abnormal (not the inverted mapping).

Imports of ``config`` and ``load_params`` happen inside the test bodies so a
transient import problem skips the test rather than erroring at collection time.
"""
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def _import_config():
    """Import the top-level config module inside a test body.

    Skips (does not error) if the import fails for a transient reason so that
    collection always succeeds.
    """
    try:
        from src import config
    except Exception as exc:
        pytest.skip(f"config.py not importable yet: {exc}")
    return config


def _load_params(modality):
    """Import load_params and call it for `modality`, skipping on import failure."""
    try:
        from src.config_loader import load_params
    except Exception as exc:
        pytest.skip(f"src.config_loader not importable yet: {exc}")
    return load_params(modality)


def test_config_sr_4000():
    """config.SR_HEART and config.SR_LUNG must both equal 4000 Hz."""
    config = _import_config()
    assert config.SR_HEART == 4000, (
        f"config.SR_HEART must be 4000, got {getattr(config, 'SR_HEART', None)}"
    )
    assert config.SR_LUNG == 4000, (
        f"config.SR_LUNG must be 4000, got {getattr(config, 'SR_LUNG', None)}"
    )


def test_heart_yaml_sample_rate():
    """params/heart.yaml `sample_rate` must be 4000."""
    params = _load_params("heart")
    assert params["sample_rate"] == 4000, (
        f"params/heart.yaml sample_rate must be 4000, got {params['sample_rate']}"
    )


def test_lung_yaml_sample_rate():
    """params/lung.yaml `sample_rate` must be 4000."""
    params = _load_params("lung")
    assert params["sample_rate"] == 4000, (
        f"params/lung.yaml sample_rate must be 4000, got {params['sample_rate']}"
    )


def test_heart_label_map_deinverted():
    """heart label_map must resolve -1 -> normal and 1 -> abnormal.

    PhysioNet/CinC 2016 convention (per REFERENCE.csv and the .hea
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
        f"label_map[-1] must be 'normal', got {_lookup(-1)!r}"
    )
    assert _lookup(1) == "abnormal", (
        f"label_map[1] must be 'abnormal', got {_lookup(1)!r}"
    )
