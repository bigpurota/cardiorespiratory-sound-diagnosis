"""
Build both patient-level train/test splits.

make_heart_splits() uses a seeded GroupShuffleSplit within databases A–E;
make_lung_splits() uses the official ICBHI split (with the 156/218 overlap repaired)
or falls back to a seeded split if the fetch fails. Each call verifies there is no
patient leakage. Outputs are written to results/splits/.

    uv run python scripts/make_splits.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config  # noqa: F401  import FIRST (seeds, paths)
from src.split import make_heart_splits, make_lung_splits


def main():
    print("[INFO] building heart split (CinC, within A–E)...")
    make_heart_splits()
    print("[INFO] building lung split (ICBHI, official+repair or fallback)...")
    make_lung_splits()
    print("[INFO] splits written to results/splits/.")


if __name__ == "__main__":
    main()
