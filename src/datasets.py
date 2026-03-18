"""
src/datasets.py — leakage-safe spectrogram Dataset + loaders (Phase 4, MODL-02).

The provider layer that turns the augmentation-free spectrogram cache
(``features/{modality}_spectrograms.npy``) into torch ``DataLoader``s for the DL training
driver, while structurally preventing the phase's #1 correctness risk — augmentation /
patient leakage. Specifically:

  - ``SpectrogramDataset(X, y, augment=False, for_effnet=False)``: SpecAugment
    (``FrequencyMasking`` + ``TimeMasking``) + light Gaussian noise are applied in
    ``__getitem__`` ONLY when ``augment=True`` (the TRAIN dataset). VAL/TEST datasets are
    built with ``augment=False`` so evaluation is never contaminated (D-05 / Pitfall 3).
    ``for_effnet=True`` routes the dB image through ``_to_effnet_image`` (min-max→[0,1] →
    1→3 channels → 224×224 → ImageNet mean/std) for the EfficientNet-B0 transfer path (D-04).
  - ``carve_val(X_train, y_train, pid_train, ...)``: carves a patient-grouped validation
    set out of the TRAIN rows with ``GroupShuffleSplit`` and re-asserts
    ``assert_no_patient_leakage`` (reused from ``src.split`` — never re-implemented). For
    the 4-class lung modality it guards class-presence (Open Question 1): if a class is
    absent from val it retries with ``test_size=0.15`` and warns about a val-loss fallback.
  - ``train_class_weights(y_train, n_classes)``: inverse-frequency class weights from the
    TRAIN labels ONLY (``compute_class_weight("balanced")``), consistent with the classical
    ``class_weight="balanced"`` (D-05).
  - ``build_loaders(cache, modality, ...)``: splits the cache by its ``split`` tag, carves
    val from train, re-asserts leakage on (train,test) AND (train,val), builds the three
    datasets (train augment=True; val/test augment=False), wraps them in seeded
    ``num_workers=0`` loaders (macOS determinism, Pitfall 6), and returns the loaders +
    train class weights + the test recording_ids (for the heart majority vote downstream).

``import config`` runs first for the SEED=42 determinism side effect.
"""
import warnings

import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

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

# ImageNet normalisation for the EfficientNet-B0 transfer path (timm default cfg, D-04).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# SpecAugment + noise defaults (04-RESEARCH §Code Examples 2, A4 conservative starts).
_FREQ_MASK_PARAM = 8   # mask up to 8 of 64 mel bins
_TIME_MASK_PARAM = 16  # mask up to 16 of 128 frames
_NOISE_SIGMA = 0.05    # dB-domain Gaussian-noise std (light)


def _to_effnet_image(spec):
    """Adapt a ``(64,128)`` dB image to EfficientNet-B0's ``(3,224,224)`` input (D-04).

    Per-image min-max scale to [0,1] (leakage-free — uses only this clip's stats), tile
    1→3 channels, bilinearly resize to 224×224, then ImageNet mean/std normalise
    (04-RESEARCH §Code Examples 3).
    """
    s = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    img = s.unsqueeze(0).repeat(3, 1, 1)  # (3,64,128)
    img = F.interpolate(
        img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)
    return (img - _IMAGENET_MEAN) / _IMAGENET_STD  # (3,224,224)


class SpectrogramDataset(Dataset):
    """Wrap an ``(N,64,128)`` log-mel cache; augment the TRAIN dataset ONLY (D-05).

    Parameters
    ----------
    X : ndarray (N, 64, 128) float32
        Log-mel dB images from the augmentation-free cache.
    y : ndarray (N,) int
        Integer class labels.
    augment : bool
        When True (TRAIN only) apply ``FrequencyMasking`` → ``TimeMasking`` → additive
        Gaussian noise in ``__getitem__``. VAL/TEST datasets MUST pass ``augment=False``.
    for_effnet : bool
        When True route the image through ``_to_effnet_image`` → ``(3,224,224)``; else
        return ``(1,64,128)`` for the small CNN.
    """

    def __init__(self, X, y, augment=False, for_effnet=False):
        self.X = np.asarray(X, dtype="float32")
        self.y = np.asarray(y, dtype=int)
        self.augment = augment
        self.for_effnet = for_effnet
        # Augmentation transforms live on the TRAIN dataset ONLY (D-05); constructing them
        # unconditionally is harmless — they are only INVOKED when self.augment is True.
        self.freq_mask = T.FrequencyMasking(freq_mask_param=_FREQ_MASK_PARAM)
        self.time_mask = T.TimeMasking(time_mask_param=_TIME_MASK_PARAM)
        self.noise_sigma = _NOISE_SIGMA

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        spec = torch.as_tensor(self.X[i], dtype=torch.float32)  # (64,128)
        if self.augment:
            # Train-only SpecAugment + light noise (gated strictly on self.augment).
            spec = self.freq_mask(spec.unsqueeze(0)).squeeze(0)
            spec = self.time_mask(spec.unsqueeze(0)).squeeze(0)
            spec = spec + torch.randn_like(spec) * self.noise_sigma
        if self.for_effnet:
            x = _to_effnet_image(spec)              # (3,224,224)
        else:
            x = spec.unsqueeze(0)                   # (1,64,128) for the small CNN
        return x, int(self.y[i])


def carve_val(X_train, y_train, pid_train, test_size=0.2, seed=42, n_classes=None):
    """Carve a patient-grouped validation set out of the TRAIN rows (leakage-free).

    Returns ``(tr_idx, va_idx)`` integer index arrays INTO the train arrays. Uses
    ``GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)`` on
    ``groups=pid_train`` so no patient spans train+val, then re-asserts
    ``assert_no_patient_leakage`` on the resulting patient-id partition.

    Open Question 1 (lung val class-presence guard): when ``n_classes`` is 4 and any class
    is absent from the val labels, retry once with ``test_size=0.15``; if a class is still
    missing, emit a warning that early stopping may need to fall back to val loss.
    """
    pid_train = np.asarray(pid_train)
    y_train = np.asarray(y_train)

    def _split(ts):
        gss = GroupShuffleSplit(n_splits=1, test_size=ts, random_state=seed)
        tr_idx, va_idx = next(gss.split(X_train, groups=pid_train))
        return tr_idx, va_idx

    tr_idx, va_idx = _split(test_size)

    # Class-presence guard for the 4-class lung modality (Open Question 1).
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

    # Reuse the canonical leakage guard (never re-implemented) on the patient partition.
    assert_no_patient_leakage(pid_train[tr_idx], pid_train[va_idx])
    return tr_idx, va_idx


def train_class_weights(y_train, n_classes):
    """Inverse-frequency class weights from the TRAIN labels ONLY (D-05).

    Mirrors the classical ``class_weight="balanced"`` so the weighted CrossEntropy is
    consistent across the comparative study. Returns a length-``n_classes`` float32 tensor.
    """
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
    """Build seeded, leakage-safe train/val/test DataLoaders from a spectrogram cache.

    Splits ``cache`` by its ``split`` tag into train/test, carves a patient-grouped val out
    of train (``carve_val``), re-asserts ``assert_no_patient_leakage`` on (train,test) AND
    (train,val), and wraps three ``SpectrogramDataset``s (train ``augment=True``; val/test
    ``augment=False``; ``for_effnet`` routed) in DataLoaders. The shuffled train loader uses
    a seeded ``torch.Generator`` and ``num_workers=0`` for macOS determinism (Pitfall 6).

    HPO knobs (additive — defaults reproduce the 04-04 behaviour exactly):
      - ``aug_strength``: scales the TRAIN SpecAugment mask params and noise sigma on the
        TRAIN dataset only (val/test remain ``augment=False`` unaffected). Default 1.0.
      - ``sampler_mode``: ``"class_weight"`` (default) returns ``class_weights`` for a
        weighted CrossEntropyLoss; ``"weighted_sampler"`` builds a seeded
        ``torch.utils.data.WeightedRandomSampler`` from TRAIN class frequencies and passes
        it to the train DataLoader (shuffle=False + sampler=…) — the two imbalance
        strategies are NEVER double-applied (the caller should use unweighted CE when
        ``sampler_mode=="weighted_sampler"``). ``class_weights`` is still returned in both
        modes for caller bookkeeping. D-05: TRAIN labels only, no SMOTE, no global scaler.

    Returns
    -------
    dict
        ``train_loader``, ``val_loader``, ``test_loader``, ``class_weights`` (from TRAIN
        labels only), ``test_recording_id`` (for the heart majority vote downstream),
        ``n_classes``, and the raw split sizes for volumetrics.
    """
    from torch.utils.data import WeightedRandomSampler  # HPO sampler_mode

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

    # (train, test) leakage re-assert at startup (logs [leakage-check OK]).
    assert_no_patient_leakage(pid_tr_all, pid_te)

    # Patient-grouped val carve out of train (also re-asserts (train,val) leakage inside).
    tr_idx, va_idx = carve_val(
        X_tr_all, y_tr_all, pid_tr_all, test_size=0.2, seed=seed, n_classes=n_classes
    )

    X_tr, y_tr = X_tr_all[tr_idx], y_tr_all[tr_idx]
    X_va, y_va = X_tr_all[va_idx], y_tr_all[va_idx]

    # TRAIN dataset with scaled augmentation params (HPO aug_strength knob).
    # aug_strength scales SpecAugment mask params + noise sigma for the TRAIN set ONLY;
    # val/test remain augment=False unchanged (D-05 / Pitfall 3).
    scaled_freq = max(1, round(_FREQ_MASK_PARAM * aug_strength))
    scaled_time = max(1, round(_TIME_MASK_PARAM * aug_strength))
    scaled_noise = float(_NOISE_SIGMA * aug_strength)

    train_ds = SpectrogramDataset(X_tr, y_tr, augment=True, for_effnet=for_effnet)
    # Override the aug params on the TRAIN dataset instance (module-level constants remain
    # unchanged so no global state is mutated — safe for concurrent subprocess runs).
    train_ds.freq_mask = T.FrequencyMasking(freq_mask_param=scaled_freq)
    train_ds.time_mask = T.TimeMasking(time_mask_param=scaled_time)
    train_ds.noise_sigma = scaled_noise

    val_ds = SpectrogramDataset(X_va, y_va, augment=False, for_effnet=for_effnet)
    test_ds = SpectrogramDataset(X_te, y_te, augment=False, for_effnet=for_effnet)

    cw = train_class_weights(y_tr, n_classes)

    gen = torch.Generator().manual_seed(seed)
    if sampler_mode == "weighted_sampler":
        # WeightedRandomSampler: per-sample weights from TRAIN inverse class frequency.
        # Built from TRAIN labels only (D-05); seed the generator for reproducibility.
        # shuffle=False because sampler handles the sampling order.
        class_freq = np.bincount(y_tr, minlength=n_classes).astype(float)
        class_freq = np.where(class_freq == 0, 1.0, class_freq)  # avoid /0
        inv_freq = 1.0 / class_freq
        sample_weights = torch.tensor([inv_freq[yi] for yi in y_tr], dtype=torch.double)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(y_tr), replacement=True, generator=gen
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=0
        )
    else:
        # Default "class_weight" path: seeded shuffle=True (reproduces 04-04 exactly).
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
