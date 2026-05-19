#!/usr/bin/env python
"""
Generate the descriptive EDA figures.

Delegates to src.eda.main and writes figures to results/figures/eda/ using the
headless matplotlib Agg backend. Reads only the manifest/cycle tables; it does not
modify splits, the manifest, or params.

    uv run python scripts/run_eda.py
"""
import sys
from pathlib import Path

# Make the project root importable regardless of the invocation cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eda import main  # noqa: E402

if __name__ == "__main__":
    main()
