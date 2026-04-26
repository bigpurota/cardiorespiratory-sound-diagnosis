# Cardiorespiratory & Arterial Sound Diagnosis

Empirical comparison of classical and deep-learning methods for diagnosing pathologies
of the cardiorespiratory system and main arteries from auscultation sound data.

## Datasets

| Dataset | Description | Recordings |
|---------|-------------|------------|
| [PhysioNet/CinC 2016](https://physionet.org/content/challenge-2016/1.0.0/) | Heart sounds (PCG), binary normal/abnormal labels | 3,126 WAV files (databases A–E) |
| [ICBHI 2017](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) | Respiratory lung sounds, 4 cycle-level classes | 920 recordings, 6,898 annotated cycles |

Arterial (carotid bruit) sounds are handled analytically — no open dataset exists;
the pipeline design is generalised to demonstrate cross-modality extensibility.

## Setup

**Prerequisites:** [uv](https://astral.sh/uv) (Astral), Python 3.11.

```bash
# Install all dependencies (creates .venv/ automatically)
uv sync
```

Or with pip (Google Colab / graders):

```bash
pip install -r requirements.txt
```

**Download datasets** (run after `uv sync`):

```bash
# Heart sounds — PhysioNet CinC 2016 (~181 MB, no authentication required)
uv run python scripts/download_heart.py

# Lung sounds — ICBHI 2017 via Kaggle API (~1.5 GB)
# Requires ~/.kaggle/kaggle.json with {"username":"...","key":"..."}, chmod 600
uv run python scripts/download_lung.py
```

## Results

[Results table will be populated in Phase 3+]

See `results/tables/` for experiment CSVs after pipeline runs.
See `results/figures/` for confusion matrices, ROC curves, and spectrograms.

## Project Structure

```
dsba_project/
├── config.py               # SEED=42, shared paths, sampling-rate constants
├── pyproject.toml          # uv project definition + pinned dependencies
├── uv.lock                 # uv exact resolver snapshot (commit this)
├── requirements.txt        # pip-compatible export for Colab/graders
├── src/                    # importable modules (Phases 2–5)
│   ├── __init__.py
│   └── config_loader.py    # load_params(modality) -> dict
├── scripts/
│   ├── download_heart.py   # PhysioNet CinC 2016 download script
│   └── download_lung.py    # ICBHI 2017 Kaggle API download script
├── data/                   # gitignored — created by download scripts
│   ├── raw/
│   │   ├── cinc2016/       # training-a/ through training-e/
│   │   └── icbhi2017/      # *.wav + *.txt annotations
│   └── processed/          # Phase 2+ feature caches (also gitignored)
├── params/
│   ├── heart.yaml          # PCG preprocessing parameters (SR, bandpass, segmentation)
│   └── lung.yaml           # Lung preprocessing parameters (SR, bandpass, cycle padding)
├── notebooks/              # Exploration notebooks (Phase 2+)
├── results/
│   ├── tables/             # Experiment result CSVs (kept for report)
│   ├── figures/            # Report figures — confusion matrices, ROC curves
│   └── splits/             # Saved patient-level split indices
└── report/                 # Typst source and compiled PDF (Phase 6+)
```

## Contributors

[Contributor table will be added in Phase 7]

11-person team — see the report for the full contributor role table (Annex 5 §2.3 format).

## Reproducibility

- **Random seed:** `SEED = 42` (set in `config.py`, imported at the top of every script)
- **Splits:** Patient-level (leakage-safe) — no recording from the same patient appears
  in both train and test
- **Dependencies:** Pinned in `requirements.txt` (produced by `uv export`); exact resolver
  snapshot in `uv.lock`. Run `uv sync` or `pip install -r requirements.txt` to reproduce.

See `config.py` for path constants and sampling-rate definitions.
