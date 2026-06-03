"""Project-scaffold tests: directory layout and params YAML"""
import pathlib

import pytest


_REQUIRED_DIRS = ["src", "data", "params", "results", "report"]


def test_dir_structure():
    """Required project directories exist."""
    missing = []
    for d in _REQUIRED_DIRS:
        if not pathlib.Path(d).is_dir():
            missing.append(d)

    assert not missing, (
        f"Missing required directories: {missing}. "
        "Create the project structure first."
    )


def test_yaml_params():
    """params/heart.yaml and params/lung.yaml are valid YAML with"""
    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed — run `uv sync` first")

    required_heart = {"sample_rate", "bandpass_low_hz", "bandpass_high_hz", "segment_length_s"}
    required_lung = {"sample_rate", "bandpass_low_hz", "bandpass_high_hz", "cycle_pad_length_s"}

    for path_str, required_keys in [
        ("params/heart.yaml", required_heart),
        ("params/lung.yaml", required_lung),
    ]:
        p = pathlib.Path(path_str)
        if not p.exists():
            pytest.skip(f"{path_str} not created yet")

        with open(p) as fh:
            data = yaml.safe_load(fh)

        if data is None:
            pytest.fail(f"{path_str} is empty — expected YAML content with keys {required_keys}")

        missing_keys = required_keys - set(data.keys())
        assert not missing_keys, (
            f"{path_str} is missing required keys: {missing_keys}"
        )
