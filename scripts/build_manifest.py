"""
scripts/build_manifest.py — thin CLI for the Phase-2 manifest builder.

Builds data/processed/manifest.csv (recording-level, one row per .wav) and the
companion data/processed/lung_cycles.csv (per-respiratory-cycle ICBHI labels) by
calling src.ingest.build_manifest(). Mirrors the structure of scripts/download_*.py.

Run with: ``uv run python scripts/build_manifest.py``
"""
import sys

from src.ingest import build_manifest


def main():
    manifest_path, cycles_path = build_manifest()
    print(f"Manifest build complete.\n  {manifest_path}\n  {cycles_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
