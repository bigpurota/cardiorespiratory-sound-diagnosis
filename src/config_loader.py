"""Utility for loading modality-specific preprocessing parameters from params/."""

import pathlib
import yaml

from src.config import PARAMS_DIR


def load_params(modality: str) -> dict:
    """Load preprocessing parameters for the given modality.

    Reads ``params/{modality}.yaml`` and returns the parsed dict.

    Parameters
    ----------
    modality:
        One of ``"heart"`` (PhysioNet/CinC 2016) or ``"lung"`` (ICBHI 2017).

    Raises
    ------
    FileNotFoundError
        When ``params/{modality}.yaml`` does not exist.
    ValueError
        When *modality* is neither ``"heart"`` nor ``"lung"``.
    """
    if modality not in ("heart", "lung"):
        raise ValueError(
            f"Unknown modality '{modality}'. Supported values: 'heart', 'lung'."
        )

    yaml_path = pathlib.Path(PARAMS_DIR) / f"{modality}.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Parameter file not found: {yaml_path}. "
            f"Expected params/{modality}.yaml relative to project root."
        )

    with open(yaml_path) as fh:
        return yaml.safe_load(fh)
