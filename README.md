# Cardiorespiratory & Arterial Sound Diagnosis

A Year-2 university term project (HSE Faculty of Computer Science, DSBA program, 2025–26).

**Research question:** Comparing machine-learning methods for diagnosing pathologies of the
cardiorespiratory system and main arteries from auscultation sound data.

**Datasets:**
- PhysioNet/CinC Challenge 2016 — Heart Sounds (PCG), 3,126 recordings
- ICBHI 2017 Respiratory Sound Database — Lung Sounds, 920 recordings

**Pipeline:** Classical ML (MFCC/log-mel + SVM/RF/XGBoost) vs. Deep Learning (CNN/EfficientNet-B0)
with patient-level leakage-safe splits and fixed random seeds.

## Setup

```bash
uv sync
```

Or with pip (Colab / graders):

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
uv run python scripts/download_heart.py   # ~181 MB
uv run python scripts/download_lung.py    # ~1.5 GB, requires ~/.kaggle/kaggle.json
uv run pytest tests/ -v
```

---

*Stub README — will be fleshed out with results tables and contributor table in Phase 7.*
