"""
tests/test_scaffold.py — DELV-01 directory structure and params YAML assertions.

Tests that:
  - Required project directories exist (src, data, params, results, report, notebooks).
    This test FAILS (not skips) if directories are absent — that is correct Wave 0
    behavior and the intent is that Wave 1 will create those dirs.
  - params/heart.yaml and params/lung.yaml are valid YAML files containing all
    required preprocessing parameter keys. These tests SKIP if the files do not
    exist yet (they are created in Wave 1).

Wave 0: test_dir_structure will FAIL (intentionally) until Wave 1 creates the dirs.
        test_yaml_params will SKIP until Wave 1 creates the YAML files.
        Collection must succeed with 0 errors.
"""
import pathlib

import pytest

# ---------------------------------------------------------------------------
# DELV-01 — directory structure
# ---------------------------------------------------------------------------
# pytest runs with cwd = project root when invoked as `pytest tests/` from root,
# so relative Path() lookups resolve correctly.

_REQUIRED_DIRS = ["src", "data", "params", "results", "report", "notebooks"]


def test_dir_structure():
    """ROADMAP success criterion 4: required project directories exist — DELV-01.

    This test FAILS (not skips) when directories are absent. That is the correct
    Wave 0 state — the failure drives Wave 1 to create the scaffold.
    """
    missing = []
    for d in _REQUIRED_DIRS:
        if not pathlib.Path(d).is_dir():
            missing.append(d)

    assert not missing, (
        f"Missing required directories: {missing}. "
        "Run the Phase 1 scaffold plan (01-01) to create the project structure."
    )


# ---------------------------------------------------------------------------
# DELV-01 — params YAML keys
# ---------------------------------------------------------------------------

def test_yaml_params():
    """params/heart.yaml and params/lung.yaml are valid YAML with required keys — DELV-01."""
    try:
        import yaml  # noqa: E501
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
            pytest.skip(f"{path_str} not created yet — Wave 0 skip; will pass after Wave 1")

        with open(p) as fh:
            data = yaml.safe_load(fh)

        if data is None:
            pytest.fail(f"{path_str} is empty — expected YAML content with keys {required_keys}")

        missing_keys = required_keys - set(data.keys())
        assert not missing_keys, (
            f"{path_str} is missing required keys: {missing_keys}"
        )
