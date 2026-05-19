"""
Cross-modal transfer and joint multi-task models for the heart/lung sound study.

Reuses the encoders, metrics, and patient-leakage checks from elsewhere in ``src``:

  - ``build_shared_encoder`` wraps the SmallCNN / EfficientNet-B0 backbone as a headless
    feature extractor.
  - ``transfer_modality`` pretrains an encoder on the source modality, swaps in a fresh
    target-sized head (handling the heart 2-class vs lung 4-class mismatch), fine-tunes on
    the target, and evaluates.
  - ``JointMultiTaskModel`` is one shared encoder with separate heart (2-class) and lung
    (4-class) heads; ``train_joint`` trains it on interleaved heart+lung batches and
    ``evaluate_joint`` scores it on both test sets.
  - ``spearman_method_rankings`` reads the classical rows of unified_comparison.csv and
    correlates the heart MAcc vs lung ICBHI rankings.

All splits are patient-level leakage-safe: ``assert_no_patient_leakage`` runs inside
``build_loaders`` and again explicitly at the top of each public entry point. No global
scaler, no SMOTE.
"""
import os

# macOS duplicate-OpenMP-runtime guard; must run before the first ``import torch``.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import time
import random as _random

from src import config  # noqa: F401 — import first for the SEED=42 side effect (determinism)

import numpy as np
import torch
import torch.nn as nn

from src.cnn import SmallCNN, build_efficientnet_b0, count_params  # noqa: F401
from src.datasets import build_loaders  # noqa: F401
from src.train_cnn import (  # noqa: F401
    train_one_model,
    evaluate,
    run_modality,
    _val_macc,
    _val_icbhi,
    _predict_test,
    HEART_LABELS,
    LUNG_LABELS,
    LUNG_NORMAL_LABEL,
)
from src.metrics import (  # noqa: F401
    majority_vote,
    heart_macc,
    icbhi_score,
    per_class_se,
    macro_f1,
    accuracy,
    save_cm,
)
from src.split import assert_no_patient_leakage  # noqa: F401

__all__ = [
    "build_shared_encoder",
    "transfer_modality",
    "JointMultiTaskModel",
    "train_joint",
    "evaluate_joint",
    "spearman_method_rankings",
]

import matplotlib
matplotlib.use("Agg")  # headless backend, must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def build_shared_encoder(arch="cnn", for_effnet=False):
    """Return a (encoder_module, feature_dim) pair for the shared cross-modal backbone.

    Parameters
    ----------
    arch : str
        "cnn"    — wraps SmallCNN.features + AdaptiveAvgPool2d(1) + Flatten into an
                   nn.Sequential; feature_dim = 128 (widths[-1] of the default SmallCNN).
        "effnet" — creates a headless EfficientNet-B0 via
                   ``timm.create_model("efficientnet_b0", pretrained=True, in_chans=3,
                   num_classes=0)``; feature_dim = m.num_features (1280 for B0).
    for_effnet : bool
        Passed through only to allow the caller to track the expected input shape.
        Encoders themselves are shape-agnostic after build_loaders adapts spectrogram
        inputs to (B,1,64,128) for the CNN or (B,3,224,224) for EffNet.

    Returns
    -------
    encoder : nn.Module
    feature_dim : int
    """
    if arch in ("effnet", "effnet_b0", "efficientnet"):
        import timm
        m = timm.create_model(
            "efficientnet_b0", pretrained=True, in_chans=3, num_classes=0
        )
        feature_dim = m.num_features
        return m, feature_dim
    else:
        # CNN: pull features + pool + flatten out of a default SmallCNN; n_classes is
        # irrelevant here since we discard the head.
        _tmp = SmallCNN(n_classes=2)
        encoder = nn.Sequential(
            _tmp.features,
            _tmp.pool,
            nn.Flatten(),
        )
        feature_dim = 128  # widths[-1] of the default (16, 32, 64, 128)
        return encoder, feature_dim


class _EncoderWithHead(nn.Module):
    """Shared encoder plus a single linear head, sized 2 for heart or 4 for lung.

    Used during the transfer_modality pretrain and fine-tune stages. Swapping the head
    while keeping the pretrained encoder is how the heart 2-class vs lung 4-class
    label-space mismatch is handled.
    """

    def __init__(self, encoder, feature_dim, n_classes, p_dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(feature_dim, n_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


def _seed_all(seed=42):
    """Set the Python, NumPy, and PyTorch RNG seeds for determinism."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def transfer_modality(
    source_cache,
    target_cache,
    source_modality,
    target_modality,
    arch="cnn",
    batch_size=32,
    max_epochs=30,
    patience=7,
    wall_cap_s=1800,
    lr=None,
    seed=42,
    out_dir=None,
):
    """Pretrain on the source modality, swap the head for the target class count, fine-tune,
    and evaluate.

    After pretraining, the source head is discarded and replaced with a fresh
    Linear(feature_dim, target_n_classes) while the pretrained encoder weights carry over
    unchanged; the encoder and new head are then fine-tuned on the target modality. Returns
    a row dict with the usual metric and volumetric fields.
    """
    _seed_all(seed)

    for_effnet = arch in ("effnet", "effnet_b0", "efficientnet")
    model_name = "effnet_b0" if for_effnet else "cnn"
    if lr is None:
        lr = 1e-4 if for_effnet else 1e-3

    out_dir = out_dir or os.path.join(FIGURES_DIR, f"transfer_{source_modality}_{target_modality}_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Source loaders (build_loaders re-asserts leakage internally).
    src_loaders = build_loaders(
        source_cache, source_modality,
        for_effnet=for_effnet, batch_size=batch_size, seed=seed,
    )
    src_n_classes = src_loaders["n_classes"]

    src_split = np.asarray(source_cache["split"])
    src_pid = np.asarray(source_cache["patient_id"])
    assert_no_patient_leakage(src_pid[src_split == "train"], src_pid[src_split == "test"])

    # Build encoder + source-sized head and pretrain on the source.
    encoder, feature_dim = build_shared_encoder(arch, for_effnet=for_effnet)
    source_model = _EncoderWithHead(encoder, feature_dim, src_n_classes)

    src_criterion = nn.CrossEntropyLoss(
        weight=src_loaders["class_weights"].to(device)
    )
    src_val_fn = _val_macc if source_modality == "heart" else _val_icbhi

    source_model, src_info = train_one_model(
        source_model,
        src_loaders["train_loader"],
        src_loaders["val_loader"],
        src_criterion,
        lr=lr,
        val_metric_fn=src_val_fn,
        max_epochs=max_epochs,
        patience=patience,
        wall_cap_s=wall_cap_s,
        device=device,
        curve_png=os.path.join(out_dir, f"curve_pretrain_{source_modality}_{model_name}.png"),
    )

    # Target loaders.
    tgt_loaders = build_loaders(
        target_cache, target_modality,
        for_effnet=for_effnet, batch_size=batch_size, seed=seed,
    )
    tgt_n_classes = tgt_loaders["n_classes"]

    tgt_split = np.asarray(target_cache["split"])
    tgt_pid = np.asarray(target_cache["patient_id"])
    assert_no_patient_leakage(tgt_pid[tgt_split == "train"], tgt_pid[tgt_split == "test"])

    # Head swap: keep the pretrained encoder, replace the head with a fresh one sized to
    # the target class count.
    transfer_model = _EncoderWithHead(
        source_model.encoder,
        feature_dim,
        tgt_n_classes,
    )

    # Fine-tune the encoder + new head on the target.
    tgt_criterion = nn.CrossEntropyLoss(
        weight=tgt_loaders["class_weights"].to(device)
    )
    tgt_val_fn = _val_macc if target_modality == "heart" else _val_icbhi

    transfer_model, tgt_info = train_one_model(
        transfer_model,
        tgt_loaders["train_loader"],
        tgt_loaders["val_loader"],
        tgt_criterion,
        lr=lr,
        val_metric_fn=tgt_val_fn,
        max_epochs=max_epochs,
        patience=patience,
        wall_cap_s=wall_cap_s,
        device=device,
        curve_png=os.path.join(out_dir, f"curve_finetune_{target_modality}_{model_name}.png"),
    )

    # Evaluate on the target test loader.
    transfer_model.eval()
    y_test, preds, win_score = _predict_test(transfer_model, tgt_loaders["test_loader"], device)

    if target_modality == "heart":
        rec_test = np.asarray(list(map(str, tgt_loaders["test_recording_id"])), dtype=object)
        pred_rec = majority_vote(preds, rec_test)
        true_rec = majority_vote(y_test, rec_test).reindex(pred_rec.index)
        y_true_rec = true_rec.to_numpy().astype(int)
        y_pred_rec = pred_rec.to_numpy().astype(int)
        m = heart_macc(y_true_rec, y_pred_rec)
        cm_png = os.path.join(out_dir, f"cm_transfer_{source_modality}to{target_modality}_{model_name}.png")
        try:
            save_cm(y_true_rec, y_pred_rec, HEART_LABELS, f"transfer {source_modality}→{target_modality}", cm_png)
        except AssertionError:
            pass  # skip a degenerate confusion matrix on tiny runs
        primary_metric_name = "MAcc"
        primary_metric = float(m["MAcc"])
        Se = float(m["Se"])
        Sp = float(m["Sp"])
        mf1 = float(m["macro_f1"])
        acc = float(m["accuracy"])
        n_train = src_loaders["n_train"]
        n_test = tgt_loaders["n_test"]
    else:
        m = icbhi_score(y_test, preds, normal_label=LUNG_NORMAL_LABEL)
        cm_png = os.path.join(out_dir, f"cm_transfer_{source_modality}to{target_modality}_{model_name}.png")
        try:
            save_cm(y_test, preds, LUNG_LABELS, f"transfer {source_modality}→{target_modality}", cm_png)
        except AssertionError:
            pass
        primary_metric_name = "ICBHI_Score"
        primary_metric = float(m["ICBHI_Score"])
        Se = float(m["Se"])
        Sp = float(m["Sp"])
        mf1 = float(macro_f1(y_test, preds))
        acc = float(accuracy(y_test, preds))
        n_train = src_loaders["n_train"]
        n_test = tgt_loaders["n_test"]

    return {
        "setting": "transfer",
        "source_modality": source_modality,
        "target_modality": target_modality,
        "model": model_name,
        "primary_metric_name": primary_metric_name,
        "primary_metric": primary_metric,
        "Se": Se,
        "Sp": Sp,
        "macro_f1": mf1,
        "accuracy": acc,
        "n_train": n_train,
        "n_test": n_test,
        "best_val_score_pretrain": src_info["best_val_score"],
        "best_val_score_finetune": tgt_info["best_val_score"],
        "epochs_ran_pretrain": src_info["epochs_ran"],
        "epochs_ran_finetune": tgt_info["epochs_ran"],
        "train_time_s": src_info["train_time_s"] + tgt_info["train_time_s"],
    }


class JointMultiTaskModel(nn.Module):
    """Shared spectrogram encoder with a 2-class heart head and a 4-class lung head.

    The encoder is a SmallCNN backbone or a headless EfficientNet-B0. ``forward(x,
    modality)`` runs the encoder then dispatches to the head for ``modality`` ("heart" or
    "lung").
    """

    def __init__(self, arch="cnn", for_effnet=False):
        super().__init__()
        encoder, feature_dim = build_shared_encoder(arch, for_effnet=for_effnet)
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.head_heart = nn.Linear(feature_dim, 2)
        self.head_lung = nn.Linear(feature_dim, 4)
        self._arch = arch

    def forward(self, x, modality):
        assert modality in {"heart", "lung"}, (
            f"JointMultiTaskModel.forward: modality must be 'heart' or 'lung', got {modality!r}"
        )
        features = self.encoder(x)
        if modality == "heart":
            return self.head_heart(features)
        return self.head_lung(features)


def train_joint(
    heart_cache,
    lung_cache,
    arch="cnn",
    batch_size=32,
    max_epochs=30,
    patience=7,
    wall_cap_s=1800,
    lr=None,
    seed=42,
    out_dir=None,
):
    """Train the JointMultiTaskModel on pooled heart+lung batches.

    Builds leakage-safe loaders for both modalities, interleaves their train batches (the
    shorter loader cycles to match the longer one), and applies per-batch CrossEntropyLoss
    with the modality's train-only class weights. Early-stops on the average of val heart
    MAcc and val lung ICBHI. Returns ``(model, {best_val_score, train_time_s, epochs_ran})``.
    """
    _seed_all(seed)

    for_effnet = arch in ("effnet", "effnet_b0", "efficientnet")
    if lr is None:
        lr = 1e-4 if for_effnet else 1e-3

    out_dir = out_dir or os.path.join(FIGURES_DIR, f"joint_{arch}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Leakage-safe loaders for both modalities.
    heart_loaders = build_loaders(
        heart_cache, "heart",
        for_effnet=for_effnet, batch_size=batch_size, seed=seed,
    )
    lung_loaders = build_loaders(
        lung_cache, "lung",
        for_effnet=for_effnet, batch_size=batch_size, seed=seed,
    )

    for cache, modality in ((heart_cache, "heart"), (lung_cache, "lung")):
        sp = np.asarray(cache["split"])
        pid = np.asarray(cache["patient_id"])
        assert_no_patient_leakage(pid[sp == "train"], pid[sp == "test"])

    model = JointMultiTaskModel(arch=arch, for_effnet=for_effnet)
    model.to(device)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    heart_crit = nn.CrossEntropyLoss(
        weight=heart_loaders["class_weights"].to(device)
    )
    lung_crit = nn.CrossEntropyLoss(
        weight=lung_loaders["class_weights"].to(device)
    )

    best_score, best_state, bad = -1.0, None, 0
    t0 = time.perf_counter()
    epoch_list = []

    for epoch in range(max_epochs):
        model.train()
        run_loss = 0.0
        seen = 0

        # Interleave heart and lung TRAIN batches deterministically (zip cycling shorter).
        heart_iter = iter(heart_loaders["train_loader"])
        lung_iter = iter(lung_loaders["train_loader"])
        h_len = len(heart_loaders["train_loader"])
        l_len = len(lung_loaders["train_loader"])
        n_batches = max(h_len, l_len)

        # Cycle whichever loader is shorter.
        import itertools
        if h_len >= l_len:
            paired = zip(
                heart_iter,
                itertools.cycle(lung_loaders["train_loader"]),
            )
        else:
            paired = zip(
                itertools.cycle(heart_loaders["train_loader"]),
                lung_iter,
            )

        capped = False
        for (xh, yh), (xl, yl) in paired:
            # Heart batch
            xh, yh = xh.to(device), yh.to(device)
            opt.zero_grad()
            out_h = model(xh, "heart")
            loss_h = heart_crit(out_h, yh)
            loss_h.backward()
            opt.step()
            run_loss += loss_h.item() * xh.size(0)
            seen += xh.size(0)

            # Lung batch
            xl, yl = xl.to(device), yl.to(device)
            opt.zero_grad()
            out_l = model(xl, "lung")
            loss_l = lung_crit(out_l, yl)
            loss_l.backward()
            opt.step()
            run_loss += loss_l.item() * xl.size(0)
            seen += xl.size(0)

            if time.perf_counter() - t0 > wall_cap_s:
                capped = True
                break

        epoch_list.append(run_loss / max(1, seen))

        # Validation: heart MAcc + lung ICBHI.
        model.eval()
        with torch.no_grad():
            # Heart
            h_pred, h_true = [], []
            for xb, yb in heart_loaders["val_loader"]:
                h_pred.append(model(xb.to(device), "heart").argmax(1).cpu())
                h_true.append(yb)
            h_pred = torch.cat(h_pred).numpy()
            h_true = torch.cat(h_true).numpy()
            val_heart = _val_macc(h_true, h_pred)

            # Lung
            l_pred, l_true = [], []
            for xb, yb in lung_loaders["val_loader"]:
                l_pred.append(model(xb.to(device), "lung").argmax(1).cpu())
                l_true.append(yb)
            l_pred = torch.cat(l_pred).numpy()
            l_true = torch.cat(l_true).numpy()
            val_lung = _val_icbhi(l_true, l_pred)

        joint_score = (val_heart + val_lung) / 2.0

        if joint_score > best_score:
            best_score = joint_score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break
        if capped or time.perf_counter() - t0 > wall_cap_s:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_info = {
        "best_val_score": float(best_score),
        "train_time_s": float(time.perf_counter() - t0),
        "epochs_ran": len(epoch_list),
    }
    return model, train_info


def evaluate_joint(
    model,
    heart_loaders,
    lung_loaders,
    out_dir=None,
    model_name="cnn",
):
    """Evaluate the JointMultiTaskModel on the heart and lung test loaders.

    Returns two row dicts: heart at recording-level MAcc and lung at cycle-level ICBHI score.
    """
    out_dir = out_dir or FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    rows = []

    # Heart test: recording-level majority vote -> MAcc.
    y_h, p_h, s_h = _predict_test(
        _ModalityAdapter(model, "heart"),
        heart_loaders["test_loader"],
        device,
    )
    rec_test_h = np.asarray(list(map(str, heart_loaders["test_recording_id"])), dtype=object)
    pred_rec_h = majority_vote(p_h, rec_test_h)
    true_rec_h = majority_vote(y_h, rec_test_h).reindex(pred_rec_h.index)
    y_true_h = true_rec_h.to_numpy().astype(int)
    y_pred_h = pred_rec_h.to_numpy().astype(int)
    mh = heart_macc(y_true_h, y_pred_h)
    cm_h = os.path.join(out_dir, f"cm_joint_heart_{model_name}.png")
    try:
        save_cm(y_true_h, y_pred_h, HEART_LABELS, f"joint heart {model_name}", cm_h)
    except AssertionError:
        pass
    rows.append({
        "setting": "joint",
        "source_modality": "heart+lung",
        "target_modality": "heart",
        "model": model_name,
        "primary_metric_name": "MAcc",
        "primary_metric": float(mh["MAcc"]),
        "Se": float(mh["Se"]),
        "Sp": float(mh["Sp"]),
        "macro_f1": float(mh["macro_f1"]),
        "accuracy": float(mh["accuracy"]),
        "n_train": heart_loaders["n_train"],
        "n_test": heart_loaders["n_test"],
    })

    # Lung test: cycle-level ICBHI score.
    y_l, p_l, _s_l = _predict_test(
        _ModalityAdapter(model, "lung"),
        lung_loaders["test_loader"],
        device,
    )
    ml = icbhi_score(y_l, p_l, normal_label=LUNG_NORMAL_LABEL)
    cm_l = os.path.join(out_dir, f"cm_joint_lung_{model_name}.png")
    try:
        save_cm(y_l, p_l, LUNG_LABELS, f"joint lung {model_name}", cm_l)
    except AssertionError:
        pass
    rows.append({
        "setting": "joint",
        "source_modality": "heart+lung",
        "target_modality": "lung",
        "model": model_name,
        "primary_metric_name": "ICBHI_Score",
        "primary_metric": float(ml["ICBHI_Score"]),
        "Se": float(ml["Se"]),
        "Sp": float(ml["Sp"]),
        "macro_f1": float(macro_f1(y_l, p_l)),
        "accuracy": float(accuracy(y_l, p_l)),
        "n_train": lung_loaders["n_train"],
        "n_test": lung_loaders["n_test"],
    })

    return rows


class _ModalityAdapter(nn.Module):
    """Wrap JointMultiTaskModel to fix modality for _predict_test compatibility."""

    def __init__(self, joint_model, modality):
        super().__init__()
        self.joint_model = joint_model
        self.modality = modality

    def forward(self, x):
        return self.joint_model(x, self.modality)


def spearman_method_rankings(unified_csv_path):
    """Compute the Spearman rank correlation between heart and lung classifier rankings.

    Reads unified_comparison.csv, selects the per-classifier classical rows at
    feature_set == "A_mfcc_delta" (heart MAcc and lung ICBHI for {logreg, svm, rf, xgb}),
    aligns them by model name, and runs ``scipy.stats.spearmanr`` over the two score vectors.
    Returns ``(rho, pvalue, n, method_labels, heart_scores, lung_scores)``.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    df = pd.read_csv(unified_csv_path)
    fs_col = "feature_set"

    heart_rows = df[
        (df[fs_col] == "A_mfcc_delta") &
        (df["primary_metric_name"] == "MAcc")
    ].set_index("model")

    lung_rows = df[
        (df[fs_col] == "A_mfcc_delta") &
        (df["primary_metric_name"] == "ICBHI_Score")
    ].set_index("model")

    common_models = sorted(set(heart_rows.index) & set(lung_rows.index))
    if not common_models:
        raise ValueError(
            f"No overlapping model names between heart/lung A_mfcc_delta rows in "
            f"{unified_csv_path}. Found heart={list(heart_rows.index)}, "
            f"lung={list(lung_rows.index)}"
        )

    heart_scores = [float(heart_rows.loc[m, "primary_metric"]) for m in common_models]
    lung_scores = [float(lung_rows.loc[m, "primary_metric"]) for m in common_models]

    result = spearmanr(heart_scores, lung_scores)
    rho = float(result.statistic) if hasattr(result, "statistic") else float(result.correlation)
    pvalue = float(result.pvalue)

    return rho, pvalue, len(common_models), common_models, heart_scores, lung_scores
