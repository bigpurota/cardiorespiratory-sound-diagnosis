#!/usr/bin/env python
"""
scripts/run_eda.py — thin CLI entry point for the Phase-2 EDA figures (D-07).

Delegates entirely to ``src.eda.main`` (logic lives in src/eda.py, not here).
Run with:

    uv run python scripts/run_eda.py

Writes the descriptive figure set to ``results/figures/eda/`` via the headless
matplotlib Agg backend. Reads only the Phase-1 manifest/cycle tables; it never
modifies splits, the manifest, or params.
"""
import sys
from pathlib import Path

# Ensure the project root is importable when run as a script (so `import config`
# and `from src.eda import main` resolve regardless of the invocation cwd).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import main  # noqa: E402

if __name__ == "__main__":
    main()
