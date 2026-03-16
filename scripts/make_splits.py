"""
scripts/make_splits.py — thin CLI: build both patient-level splits (DATA-03).

Runs make_heart_splits() (seeded GroupShuffleSplit within A–E, D-10) and
make_lung_splits() (official ICBHI fetch + 156/218 repair, else seeded fallback;
D-03/D-04). Each call logs its own ``[leakage-check OK]`` line (surfaces in Methods).
Outputs land in results/splits/ (config.SPLITS_DIR, git-committed).

    uv run python scripts/make_splits.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: F401  import FIRST (seeds, paths)
from src.split import make_heart_splits, make_lung_splits


def main():
    print("[INFO] building heart split (CinC, within A–E)...")
    make_heart_splits()
    print("[INFO] building lung split (ICBHI, official+repair or fallback)...")
    make_lung_splits()
    print("[INFO] splits written to results/splits/.")


if __name__ == "__main__":
    main()
