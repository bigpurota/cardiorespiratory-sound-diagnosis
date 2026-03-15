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
torch.backends.cudnn.deterministic = True  # eliminates cuDNN non-determinism
torch.backends.cudnn.benchmark = False     # disables auto-tuner (would pick different algos)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
PARAMS_DIR     = os.path.join(PROJECT_ROOT, "params")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
SPLITS_DIR     = os.path.join(PROJECT_ROOT, "results", "splits")

CINC2016_DIR   = os.path.join(DATA_RAW, "cinc2016")
ICBHI2017_DIR  = os.path.join(DATA_RAW, "icbhi2017")

# ── Sampling rates (canonical, used by all preprocessing) ────────────────────
SR_HEART = 2000   # PhysioNet/CinC 2016 native rate; consistent with literature
SR_LUNG  = 8000   # ICBHI 2017 target resample rate; recommended by literature
