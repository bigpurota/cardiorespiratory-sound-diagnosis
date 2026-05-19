"""
Build the recording-level manifest and the per-cycle ICBHI label table.

Calls src.ingest.build_manifest() to write data/processed/manifest.csv (one row
per .wav) and data/processed/lung_cycles.csv (one row per respiratory cycle).

    uv run python scripts/build_manifest.py
"""
import sys

from src.ingest import build_manifest


def main():
    manifest_path, cycles_path = build_manifest()
    print(f"Manifest build complete.\n  {manifest_path}\n  {cycles_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
