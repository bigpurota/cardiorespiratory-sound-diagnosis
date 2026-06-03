"""Spectrogram Dataset and DataLoaders for the CNN training"""
import warnings

from src import config

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from src.split import assert_no_patient_leakage

__all__ = [
    "SpectrogramDataset",
    "carve_val",
    "train_class_weights",
    "build_loaders",
]

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_FREQ_MASK_PARAM = 8
_TIME_MASK_PARAM = 16
_NOISE_SIGMA = 0.05


def _to_effnet_image(spec):
    """Adapt a ``(64,128)`` dB image to EfficientNet-B0's"""
    s = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    img = s.unsqueeze(0).repeat(3, 1, 1)
    img = F.interpolate(
        img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    return (img - _IMAGENET_MEAN) / _IMAGENET_STD


class SpectrogramDataset(Dataset):
    """Wrap an ``(N,64,128)`` log-mel cache; augmentation applies"""

    def __init__(self, X, y, augment=False, for_effnet=False):
        self.X = np.asarray(X, dtype="float32")
        self.y = np.asarray(y, dtype=int)
        self.augment = augment
        self.for_effnet = for_effnet
        self.freq_mask = T.FrequencyMasking(freq_mask_param=_FREQ_MASK_PARAM)
        self.time_mask = T.TimeMasking(time_mask_param=_TIME_MASK_PARAM)
        self.noise_sigma = _NOISE_SIGMA

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        spec = torch.as_tensor(self.X[i], dtype=torch.float32)
        if self.augment:
            spec = self.freq_mask(spec.unsqueeze(0)).squeeze(0)
            spec = self.time_mask(spec.unsqueeze(0)).squeeze(0)
            spec = spec + torch.randn_like(spec) * self.noise_sigma
        if self.for_effnet:
            x = _to_effnet_image(spec)
        else:
            x = spec.unsqueeze(0)
        return x, int(self.y[i])


def carve_val(X_train, y_train, pid_train, test_size=0.2, seed=42, n_classes=None):
    """Carve a patient-grouped validation set out of the train"""
    pid_train = np.asarray(pid_train)
    y_train = np.asarray(y_train)

    def _split(ts):
        gss = GroupShuffleSplit(n_splits=1, test_size=ts, random_state=seed)
        tr_idx, va_idx = next(gss.split(X_train, groups=pid_train))
        return tr_idx, va_idx

    tr_idx, va_idx = _split(test_size)

    if n_classes is not None and n_classes > 2:
        present = set(np.unique(y_train[va_idx]).tolist())
        if len(present) < n_classes:
            tr_idx2, va_idx2 = _split(0.15)
            present2 = set(np.unique(y_train[va_idx2]).tolist())
            if len(present2) >= len(present):
                tr_idx, va_idx = tr_idx2, va_idx2
                present = present2
            if len(present) < n_classes:
                warnings.warn(
                    "carve_val: not all classes present in the patient-grouped val carve "
                    f"(present={sorted(present)}, expected {n_classes}); early stopping may "
                    "need to fall back to val loss instead of val ICBHI Score.",
                    RuntimeWarning,
                )

    assert_no_patient_leakage(pid_train[tr_idx], pid_train[va_idx])
    return tr_idx, va_idx


def train_class_weights(y_train, n_classes):
    """Inverse-frequency class weights from the train labels only."""
    classes = np.arange(n_classes)
    w = compute_class_weight("balanced", classes=classes, y=np.asarray(y_train))
    return torch.tensor(w, dtype=torch.float32)


def build_loaders(
    cache,
    modality,
    for_effnet=False,
    batch_size=32,
    seed=42,
    aug_strength=1.0,
    sampler_mode="class_weight",
):
    """Build seeded, leakage-safe train/val/test DataLoaders from"""
    from torch.utils.data import WeightedRandomSampler

    X = np.asarray(cache["X"], dtype="float32")
    y = np.asarray(cache["labels"], dtype=int)
    pid = np.asarray(cache["patient_id"])
    split = np.asarray(cache["split"])
    rec_id = np.asarray(cache["recording_id"])

    n_classes = 2 if modality == "heart" else 4

    is_train = split == "train"
    is_test = split == "test"

    X_tr_all, y_tr_all, pid_tr_all = X[is_train], y[is_train], pid[is_train]
    X_te, y_te, pid_te = X[is_test], y[is_test], pid[is_test]
    rec_te = rec_id[is_test]

    assert_no_patient_leakage(pid_tr_all, pid_te)

    tr_idx, va_idx = carve_val(
        X_tr_all, y_tr_all, pid_tr_all, test_size=0.2, seed=seed, n_classes=n_classes
    )

    X_tr, y_tr = X_tr_all[tr_idx], y_tr_all[tr_idx]
    X_va, y_va = X_tr_all[va_idx], y_tr_all[va_idx]

    scaled_freq = max(1, round(_FREQ_MASK_PARAM * aug_strength))
    scaled_time = max(1, round(_TIME_MASK_PARAM * aug_strength))
    scaled_noise = float(_NOISE_SIGMA * aug_strength)

    train_ds = SpectrogramDataset(X_tr, y_tr, augment=True, for_effnet=for_effnet)
    train_ds.freq_mask = T.FrequencyMasking(freq_mask_param=scaled_freq)
    train_ds.time_mask = T.TimeMasking(time_mask_param=scaled_time)
    train_ds.noise_sigma = scaled_noise

    val_ds = SpectrogramDataset(X_va, y_va, augment=False, for_effnet=for_effnet)
    test_ds = SpectrogramDataset(X_te, y_te, augment=False, for_effnet=for_effnet)

    cw = train_class_weights(y_tr, n_classes)

    gen = torch.Generator().manual_seed(seed)
    if sampler_mode == "weighted_sampler":
        class_freq = np.bincount(y_tr, minlength=n_classes).astype(float)
        class_freq = np.where(class_freq == 0, 1.0, class_freq)
        inv_freq = 1.0 / class_freq
        sample_weights = torch.tensor([inv_freq[yi] for yi in y_tr], dtype=torch.double)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(y_tr), replacement=True, generator=gen
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=gen, num_workers=0
        )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": cw,
        "test_recording_id": rec_te,
        "n_classes": n_classes,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_te)),
    }
