"""Shared configuration for the cardiorespiratory sound ML"""
import random
import os
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
PARAMS_DIR     = os.path.join(PROJECT_ROOT, "params")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")
SPLITS_DIR     = os.path.join(PROJECT_ROOT, "results", "splits")

CINC2016_DIR   = os.path.join(DATA_RAW, "cinc2016")
ICBHI2017_DIR  = os.path.join(DATA_RAW, "icbhi2017")

SR_HEART = 4000
SR_LUNG  = 4000
