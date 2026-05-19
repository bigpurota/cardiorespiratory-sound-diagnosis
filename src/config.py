"""
Shared configuration for the cardiorespiratory sound ML pipeline.
Import this module FIRST in every script to guarantee reproducibility.
"""
import random
import os
import numpy as np
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)           # no-op if no CUDA
torch.backends.cudnn.deterministic = True  # eliminate cuDNN non-determinism
torch.backends.cudnn.benchmark = False     # disable the auto-tuner (varies algos)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
PARAMS_DIR     = os.path.join(PROJECT_ROOT, "params")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
SPLITS_DIR     = os.path.join(PROJECT_ROOT, "results", "splits")

CINC2016_DIR   = os.path.join(DATA_RAW, "cinc2016")
ICBHI2017_DIR  = os.path.join(DATA_RAW, "icbhi2017")

# ── Sampling rates (used by all preprocessing) ───────────────────────────────
# A common 4000 Hz target for both modalities so heart and lung share one feature
# space. Heart is upsampled 2000 -> 4000; lung is downsampled from mixed native
# rates. Nyquist (2000 Hz) covers the lung 1800 Hz cutoff and the full heart band.
SR_HEART = 4000   # upsampled from PhysioNet/CinC 2016 native 2000 Hz
SR_LUNG  = 4000   # downsampled from mixed ICBHI native rates to 4000 Hz
