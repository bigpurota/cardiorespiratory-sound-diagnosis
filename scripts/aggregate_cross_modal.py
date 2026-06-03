"""Aggregate per-seed cross-modal battery JSONs into mean+/-std CSVs.

Standalone aggregation for the case where the multiseed orchestrator was
interrupted but all per-(seed, arch) worker JSONs exist in results/_cm_tmp/.
Reuses the orchestrator's _aggregate so the output matches the normal path.
"""
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.run_cross_modal_multiseed import _aggregate, SUMMARY_CSV, RAW_CSV

TMP = PROJECT_ROOT / "results" / "_cm_tmp"


def main():
    rows = []
    for f in sorted(glob.glob(str(TMP / "cm_seed*.json"))):
        data = json.loads(Path(f).read_text())
        rows.extend(data)
        print(f"  loaded {len(data)} rows from {Path(f).name}")
    if not rows:
        print("no json rows found")
        sys.exit(1)

    raw = pd.DataFrame(rows)
    raw.to_csv(RAW_CSV, index=False)
    print(f"[wrote] {RAW_CSV} ({len(raw)} rows)")

    summary = _aggregate(raw)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[wrote] {SUMMARY_CSV} ({len(summary)} rows)")

    print("\n=== CROSS-MODAL MULTI-SEED SUMMARY ===")
    for _, r in summary.iterrows():
        print(f"  {r['setting']:9s} {r['source_modality']:10s}->{r['target_modality']:5s} "
              f"{r['model']:10s} {r['primary_metric_name']:11s}="
              f"{r['primary_metric_mean']:.4f}+/-{r['primary_metric_std']:.4f} (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
