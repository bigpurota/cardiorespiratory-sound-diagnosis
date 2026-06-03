"""Build both patient-level train/test splits."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import config
from src.split import make_heart_splits, make_lung_splits


def main():
    print("[INFO] building heart split (CinC, within A–E)...")
    make_heart_splits()
    print("[INFO] building lung split (ICBHI, official+repair or fallback)...")
    make_lung_splits()
    print("[INFO] splits written to results/splits/.")


if __name__ == "__main__":
    main()
