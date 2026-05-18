# Cardiorespiratory & Arterial Sound Diagnosis

A reproducible, leakage-safe **comparative study** of classical feature-engineering methods
(MFCC + SVM / Random Forest / XGBoost / Logistic Regression) versus deep learning
(custom CNN, EfficientNet-B0, Audio Spectrogram Transformer) for diagnosing pathologies of
the cardiorespiratory system and main arteries from **auscultation sound** — heart sounds
(phonocardiograms), lung sounds, and, analytically, arterial (carotid bruit) sounds.

HSE Faculty of Computer Science · DSBA / Applied Data Analysis (ПАД), Year 2, 2025–26 ·
research project. The accompanying report is authored in Typst → PDF (`report/`).

## Headline results

Primary metric: **MAcc = (Se + Sp) / 2** for heart (binary); **ICBHI Score = (Se + Sp) / 2**
for lung (4-class). Deep-learning rows are tuned (128-trial val-only HPO) and reported as
multi-seed mean ± std over seeds {1, 2, 42}; classical rows are single deterministic runs.

| Modality | Family | Best model | Primary metric |
|----------|--------|------------|----------------|
| Heart (CinC 2016) | Classical | XGBoost (feat. set B) | **MAcc 0.903** |
| Heart (CinC 2016) | Deep | EfficientNet-B0 | **MAcc 0.898 ± 0.008** |
| Lung (ICBHI 2017) | Classical | SVM (feat. set B) | **ICBHI 0.537** |
| Lung (ICBHI 2017) | Deep | EfficientNet-B0 | **ICBHI 0.555 ± 0.016** |

**Key findings**

1. **After equal HPO effort, the classical-vs-deep gap on heart closes to within noise**
   (XGBoost 0.903 ≈ EfficientNet-B0 0.898 ± 0.008). Lung is hard for *every* family (~0.54–0.56).
2. **Method rankings transfer only partially across modalities** — XGBoost dominates heart but
   falls to third on lung; SVM is uniformly competitive. Spearman ρ = 0.60 (p = 0.40, n = 4).
3. **Deep cross-modal transfer is strongly asymmetric**: lung→heart transfers near in-domain
   (MAcc 0.854 / 0.876), heart→lung is weak/negative (ICBHI 0.524 / 0.526).
4. **AST honest limitation**: the pretrained Audio Spectrogram Transformer is fully integrated
   but collapses to the majority class within the available fine-tune budget — recorded as a
   documented non-convergence (no fabricated number), presented as a methodological extension.
5. **Arterial sounds** are treated analytically — no open carotid-bruit dataset exists; the
   pipeline is shown to generalise in principle (Chapter 4 sub-study).

All numbers live in `results/tables/unified_comparison.csv` (20 rows) and the
`cross_modal_*` / `metrics_*` CSVs; figures in `results/figures/`.

## Datasets

| Dataset | Description | Size |
|---------|-------------|------|
| [PhysioNet/CinC 2016](https://physionet.org/content/challenge-2016/1.0.0/) | Heart sounds (PCG), binary normal/abnormal | ≈3,240 recordings (databases A–E) |
| [ICBHI 2017](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge) | Respiratory lung sounds, 4 cycle-level classes | 920 recordings, 6,898 cycles |

Arterial (carotid bruit) sounds are handled analytically — no open dataset exists (Chapter 4).

## Setup

**Prerequisites:** [uv](https://astral.sh/uv) (Astral), Python 3.11.

```bash
uv sync                          # creates .venv/ and installs the pinned stack
# or, for Colab / graders:
pip install -r requirements.txt
```

**Download datasets** (after `uv sync`):

```bash
uv run python scripts/download_heart.py    # PhysioNet CinC 2016 (~181 MB, no auth)
uv run python scripts/download_lung.py     # ICBHI 2017 via Kaggle API (~1.5 GB)
uv run python scripts/fetch_icbhi_split.py # official 60/40 patient-independent split
```

## Reproduce the study (end-to-end)

The pipeline is configuration-driven (`config.py` sets `SEED = 42` and all paths) and runs
identically on both modalities. Classical experiments are CPU-fast; the deep-learning steps
are CPU-feasible at small scale but were run on GPU (see `notebooks/run_cnn_gpu.ipynb`).

```bash
# 1. Ingest + manifest + leakage-safe grouped splits (patient-level lung / recording-level heart)
uv run python scripts/build_manifest.py
uv run python scripts/make_splits.py
uv run python scripts/run_eda.py

# 2. Classical arm — MFCC/Δ (+ spectral) features → SVM / RF / XGBoost / LogReg
uv run python scripts/build_features.py
uv run python scripts/run_classical.py      # → metrics_{heart,lung}_classical.csv + unified rows

# 3. Deep-learning arm — log-mel spectrograms → SmallCNN + EfficientNet-B0
uv run python scripts/build_spectrograms.py
uv run python scripts/run_cnn.py            # core DL rows + learning-curve/CM figures
uv run python scripts/run_hpo.py            # 128-trial val-only hyperparameter search (GPU)
uv run python scripts/run_multiseed.py      # 3-seed mean ± std → hardens the 4 DL rows
uv run python scripts/run_ast.py            # AST fine-tune (honest non-convergence record)

# 4. Novelty — deep cross-modal transfer + joint multi-task + Spearman
uv run python scripts/run_cross_modal.py --arch both

# 5. Tests
uv run pytest -q
```

## Project structure

```
dsba_project/
├── config.py                # SEED=42, shared paths, sampling-rate constants
├── pyproject.toml / uv.lock # pinned dependencies (commit both)
├── requirements.txt         # pip export for Colab/graders
├── src/                     # importable pipeline modules
│   ├── ingest.py preprocess.py segment.py split.py   # data → leakage-safe splits
│   ├── features.py spectrograms.py datasets.py       # MFCC + log-mel feature paths
│   ├── train_classical.py train_cnn.py               # classical + deep training/eval
│   ├── cnn.py ast_model.py cross_modal.py            # models: SmallCNN/EffNet, AST, transfer
│   ├── metrics.py           # MAcc / ICBHI Score, majority-vote, confusion matrices
│   └── eda.py config_loader.py
├── scripts/                 # CLI drivers (download → features → experiments)
│   ├── download_*.py fetch_icbhi_split.py build_*.py make_splits.py
│   └── run_classical.py run_cnn.py run_hpo.py run_multiseed.py run_ast.py run_cross_modal.py
├── tests/                   # leakage / determinism / metrics / pipeline test suite
├── params/                  # heart.yaml / lung.yaml preprocessing parameters
├── results/
│   ├── tables/              # experiment CSVs (unified_comparison.csv + per-arm)
│   └── figures/             # confusion matrices, learning curves, cross-modal heatmap
├── report/                  # Typst source (sections/ + refs.bib) → main.pdf
├── notebooks/               # EDA + GPU entry (run_cnn_gpu.ipynb)
├── reference/               # course annexes + title-page templates (see reference/README.md)
├── data/                    # gitignored — created by the download scripts
└── repo_link.txt            # submission repository link (TXT deliverable)
```

## Reproducibility guarantees

- **Random seed:** `SEED = 42` (set in `config.py`, imported at the top of every script);
  deep rows additionally averaged over seeds {1, 2, 42}.
- **Leakage-safe grouped splits:** patient-level for lung (ICBHI's official 60/40
  patient-independent split); recording-level for heart (CinC 2016 ships no
  recording→subject map, so grouping is by recording — this still eliminates the
  window-level leakage that dominates windowed audio). `src/split.py` asserts zero
  overlap (`[leakage-check OK]`) on (train, test) **and** (train, val).
- **No leakage shortcuts:** no global scaler (scaling inside CV-fold pipelines only), no SMOTE,
  no test-set model selection (HPO selects on the validation split only).
- **Pinned dependencies:** `uv.lock` (exact) + `requirements.txt` (pip export). `uv sync` reproduces.
- **Honest reporting:** non-converged / weak / negative results are recorded as real numbers,
  never fabricated or hidden.

## Report

The research report is authored in **Typst** and compiled to PDF (Annex 5 structure, Annex 7
formatting). Build from the repo root (the `--root` flag is required — figures reference
`../../results/...`):

```bash
typst compile --root . report/main.typ report/main.pdf
```

## Author & context

Individual research project (project-group format, one member). Author: **Tsember Andrei
Alekseevich** (Цембер Андрей Алексеевич), group **БПАД244**. Scientific supervisor:
**Tomashchuk Kornei Kirillovich** (Lecturer). HSE FCS, DSBA / Applied Data Analysis, 2025–26.
Full contributor and methodology details are in the report (Annex 5 format).

## License

Code in this repository is released under the **MIT License** (see [`LICENSE`](LICENSE)).
The datasets are **not** redistributed here and remain under their own terms —
PhysioNet/CinC 2016 (Open Data Commons Attribution v1.0) and ICBHI 2017 (open for
research use; cite Rocha et al. 2019). Download them with the scripts in `scripts/`.
