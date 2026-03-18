"""
src/train_cnn.py — leakage-safe DL training & evaluation entry point (Phase 4, MODL-02).

The train+evaluate half of the Phase-4 deep-learning comparative study. Mirrors
``src/train_classical.py::run_experiments`` (the fit→predict→evaluate→row-dict flow) so
the DL rows are byte-comparable with the classical rows, and copies its heart
recording-level aggregation block verbatim (swapping ``estimator.predict``→CNN argmax and
``_abnormal_proba``→softmax column-1). Everything numeric flows through ``src/metrics.py``
(no re-implemented metrics):

  - ``train_one_model(model, train_loader, val_loader, criterion, lr, val_metric_fn, ...)``
    — a device-auto-detect (CPU here / CUDA on the funded GPU) loop with Adam over the
    ``requires_grad`` params, weighted CrossEntropy, ≤``max_epochs`` epochs, early stop on
    the val primary metric (val MAcc heart / val ICBHI lung) with ``patience``, a wall-clock
    cap (D-03 deadline protection), best-checkpoint restore, and a learning-curve PNG
    (train vs val loss). Returns ``(model, {best_val_score, train_time_s, epochs_ran})``.
  - ``evaluate(model, test_loader, modality, rec_test, ...)`` — predicts on TEST and
    evaluates heart at RECORDING level (majority vote → MAcc, §Pattern 5) / lung at CYCLE
    level (ICBHI Score, §Pattern 6), saves a non-degenerate confusion-matrix figure, and
    returns the Phase-3-shaped metric+volumetric row dict PLUS a ``params`` field (D-09).
  - ``run_modality(cache, modality, model, ...)`` — the single per-experiment driver the
    Plan-04 CSV script calls: ``build_loaders`` → build model → ``train_one_model`` →
    ``evaluate`` → row dict. Accepts ``lr`` (CNN 1e-3 / EffNet 1e-4) and ``batch_size`` as
    params; writes PNGs/checkpoints only (NO CSV — that is scripts/run_cnn.py, Plan 04).

``import config`` runs first for the SEED=42 determinism side effect.
"""
import os

# macOS duplicate-OpenMP-runtime guard (copied VERBATIM from train_classical.py): torch
# (and any sklearn/xgboost loaded alongside) each bundle their own libomp.dylib; capping
# OpenMP to a single team and allowing the duplicate runtime prevents the collision
# segfault. MUST run BEFORE the first ``import torch``. ``setdefault`` lets a caller override.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import copy

import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # MUST precede pyplot import — headless (Pitfall 7)
import matplotlib.pyplot as plt  # noqa: E402

from src.metrics import (  # noqa: E402 — REUSE; never re-implemented
    majority_vote,
    heart_macc,
    icbhi_score,
    per_class_se,
    macro_f1,
    accuracy,
    save_cm,
)
from src.split import assert_no_patient_leakage  # noqa: E402
from src.datasets import build_loaders  # noqa: E402
from src.cnn import SmallCNN, build_efficientnet_b0, count_params  # noqa: E402

__all__ = ["train_one_model", "evaluate", "run_modality"]

# Label constants (mirror src/train_classical.py).
HEART_LABELS = [0, 1]
LUNG_LABELS = [0, 1, 2, 3]
LUNG_NORMAL_LABEL = 3  # {crackle:0, wheeze:1, both:2, normal:3}

RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# ---------------------------------------------------------------------------
# val-metric wrappers: adapt the metric-dict helpers to (y_true, y_pred)->scalar
# (the signature train_one_model's early-stop monitor expects). REUSE only.
# ---------------------------------------------------------------------------
def _val_macc(y_true, y_pred):
    """Heart early-stop monitor: recording/window-level MAcc (cycle-free) — REUSE heart_macc."""
    return float(heart_macc(np.asarray(y_true), np.asarray(y_pred))["MAcc"])


def _val_icbhi(y_true, y_pred):
    """Lung early-stop monitor: cycle-level ICBHI Score — REUSE icbhi_score."""
    return float(
        icbhi_score(np.asarray(y_true), np.asarray(y_pred), normal_label=LUNG_NORMAL_LABEL)[
            "ICBHI_Score"
        ]
    )


# ---------------------------------------------------------------------------
# Precise-BN: recompute BatchNorm running stats so eval matches the trained weights
# ---------------------------------------------------------------------------
def _recalibrate_batchnorm(model, train_loader, device, n_passes=3, max_batches=None):
    """Recompute BatchNorm running mean/var as an exact average over the train loader.

    Fixes the train/eval BatchNorm-statistic mismatch that, on small batches / few epochs,
    collapses the eval-mode forward to a near-constant output (degenerate predictions). Each
    module's running stats are reset and ``momentum=None`` (cumulative moving average), then
    a handful of no-grad TRAIN-MODE forward passes accumulate the exact dataset statistics.
    No gradients are taken and no weights change — only the running buffers are recalibrated.
    A no-op when the model has no BatchNorm layers.

    Deadline/efficiency guards (Rule 1 fix):
      - **Frozen-backbone skip:** when the BatchNorm layers are FROZEN
        (``weight.requires_grad is False`` — the EfficientNet head-only fallback path, D-04),
        their running stats were never disturbed by training (the backbone never updated), so
        recomputing them is a no-benefit operation that, on the 33k-window heart EffNet, costs
        ~3 full forward passes over the whole train set (tens of minutes on CPU). Skip it.
      - **Pass batch cap:** ``max_batches`` bounds the number of batches per pass so the
        recalibration cannot dominate the wall-clock cap on large datasets; a few hundred
        batches already give an accurate running-stat estimate.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    # Frozen-backbone skip: if the BN affine params are frozen, the backbone did not train,
    # so its pretrained running stats are already correct — recomputing wastes compute.
    trainable_bn = [
        m for m in bns
        if getattr(m, "weight", None) is not None and m.weight.requires_grad
    ]
    if not trainable_bn:
        return  # all BN layers frozen (e.g. EffNet head-only freeze, D-04) → no-op
    saved_momentum = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average → exact dataset mean/var
    was_training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(max(1, n_passes)):
            for i, (xb, _yb) in enumerate(train_loader):
                if max_batches is not None and i >= max_batches:
                    break  # bound the recalibration cost on large datasets
                model(xb.to(device))
    for m, mom in zip(bns, saved_momentum):
        m.momentum = mom
    if not was_training:
        model.eval()


# ---------------------------------------------------------------------------
# §Code Examples 6 — training loop: device auto-detect, early stop, wall-cap, curve
# ---------------------------------------------------------------------------
def train_one_model(
    model,
    train_loader,
    val_loader,
    criterion,
    lr,
    val_metric_fn,
    max_epochs=30,
    patience=7,
    wall_cap_s=1800,
    ckpt_path=None,
    curve_png=None,
    device=None,
    weight_decay=0.0,
):
    """Train ``model`` with early stop + wall cap; restore best by val metric; write curve PNG.

    ``device`` auto-detects CUDA→CPU (D-07) so the SAME code runs on the funded GPU and the
    CPU fallback. Adam optimises only the ``requires_grad`` params (so the EfficientNet
    head-only freeze path trains just the classifier). Each epoch accumulates a train loss,
    then a val loss + val preds; ``score = val_metric_fn(true, pred)`` is the val primary
    metric (val MAcc heart / val ICBHI lung, D-06). The best ``state_dict`` (by score) is
    deep-copied; the loop early-stops after ``patience`` non-improving epochs and hard-breaks
    when the wall-clock cap ``wall_cap_s`` is exceeded (D-03). Returns
    ``(model, {"best_val_score", "train_time_s", "epochs_ran"})``.

    HPO knob: ``weight_decay`` (default 0.0 → same Adam as 04-04; any positive value adds L2
    regularisation).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
        weight_decay=weight_decay,
    )

    best_score, best_state, bad = -1.0, None, 0
    tr_losses, va_losses = [], []
    t0 = time.perf_counter()

    capped_mid_epoch = False
    for _epoch in range(max_epochs):
        # ---- train ----
        model.train()
        run = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
            seen += xb.size(0)
            # MID-EPOCH wall-cap check (Rule 1/D-03 deadline protection): on a large
            # dataset a single epoch can vastly exceed wall_cap_s, so the end-of-epoch cap
            # alone is ineffective (the loop cannot interrupt an in-flight epoch). Break out
            # of the train loop here, then still run ONE validation pass below so the
            # partially-trained epoch contributes a val score + best_state (a non-degenerate
            # model) rather than wasting the compute. This makes the cap honest for the 33k
            # heart-window EfficientNet path where one CPU epoch can take ~40+ min.
            if time.perf_counter() - t0 > wall_cap_s:
                capped_mid_epoch = True
                break
        tr_losses.append(run / max(1, seen))

        # ---- validation: loss + primary metric (val MAcc heart / val ICBHI lung) ----
        model.eval()
        vloss = 0.0
        vpred, vtrue = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                vloss += criterion(out, yb.to(device)).item() * xb.size(0)
                vpred.append(out.argmax(1).cpu())
                vtrue.append(yb)
        va_losses.append(vloss / len(val_loader.dataset))
        score = val_metric_fn(
            torch.cat(vtrue).numpy(), torch.cat(vpred).numpy()
        )

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break  # early stop (D-06)
        if capped_mid_epoch or time.perf_counter() - t0 > wall_cap_s:
            break  # wall-clock cap (D-03 deadline protection; mid- or end-of-epoch)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Precise-BN recalibration (Rule 1 fix): with small batches / few epochs the BatchNorm
    # *running* statistics used at eval time lag the per-batch statistics the weights were
    # trained under, so eval-mode forward can collapse to a near-constant (degenerate)
    # output until the running stats warm up over many epochs. Recompute the running
    # mean/var as an exact average over the (augmentation-free style) train pass so the
    # eval-time normalisation matches the learned weights. This is the standard "precise BN"
    # technique (Ioffe 2017 / He et al.) — it improves eval stability on the real runs too,
    # and is what makes the 2-epoch smoke run produce a non-degenerate confusion matrix.
    # Cap precise-BN to ~200 batches/pass so it cannot dominate the wall-clock budget on the
    # 33k-window heart set (a few hundred batches give an accurate running-stat estimate);
    # frozen-backbone EffNet is skipped entirely inside the helper (Rule 1 deadline fix).
    _recalibrate_batchnorm(model, train_loader, device, max_batches=200)

    if ckpt_path:
        torch.save(model.state_dict(), ckpt_path)
    if curve_png:
        os.makedirs(os.path.dirname(os.path.abspath(curve_png)), exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(tr_losses, label="train loss")
        ax.plot(va_losses, label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        ax.set_title(os.path.basename(curve_png))
        fig.tight_layout()
        fig.savefig(curve_png, dpi=150)
        plt.close(fig)

    return model, {
        "best_val_score": float(best_score),
        "train_time_s": float(time.perf_counter() - t0),
        "epochs_ran": len(tr_losses),
    }


# ---------------------------------------------------------------------------
# prediction helper: per-batch TEST argmax + softmax column-1 (heart abnormal score)
# ---------------------------------------------------------------------------
def _predict_test(model, test_loader, device):
    """Run TEST inference; return (y_true, y_pred, abnormal_score) numpy arrays.

    ``abnormal_score`` is softmax column-1 (P(abnormal)) — the CNN analogue of
    classical ``_abnormal_proba`` used for the heart recording-level AUC (§Pattern 3).
    """
    model.eval()
    y_true, y_pred, score1 = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            y_pred.append(out.argmax(1).cpu().numpy())
            y_true.append(np.asarray(yb))
            # softmax column-1 = P(abnormal) for the binary heart head.
            score1.append(prob[:, 1] if prob.shape[1] > 1 else prob[:, 0])
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(score1),
    )


# ---------------------------------------------------------------------------
# §Pattern 5/6/7 — evaluation: heart recording-level MAcc / lung cycle-level ICBHI
# ---------------------------------------------------------------------------
def evaluate(
    model,
    test_loader,
    modality,
    rec_test,
    figures_dir,
    fs_label,
    model_name,
    params_count,
    train_time_s,
    volumetrics,
    device=None,
):
    """Evaluate ``model`` on TEST and return the Phase-3-shaped metric+volumetric row dict.

    HEART — copies the train_classical.py recording-level aggregation VERBATIM (swapping
    ``estimator.predict``→CNN argmax and ``_abnormal_proba``→softmax column-1): per-window
    preds are reduced to recording level via ``majority_vote``, the per-recording abnormal
    score is the mean softmax-col1, and ``heart_macc`` yields the MAcc suite + AUC. LUNG —
    cycle-level ``icbhi_score`` + ``per_class_se`` (no aggregation). Both write a
    non-degenerate ``save_cm`` figure. Returns the row dict (metric suite + volumetrics +
    ``params``) plus the raw ``test_true``/``test_pred`` arrays (for the smoke CM contract).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(figures_dir, exist_ok=True)

    y_test, preds, win_score = _predict_test(model, test_loader, device)

    if modality == "heart":
        # Recording-level majority vote → MAcc (§Pattern 5). VERBATIM from train_classical.py.
        rec_test = np.asarray(list(map(str, rec_test)), dtype=object)
        pred_rec = majority_vote(preds, rec_test)  # Series idx=recording_id
        true_rec = majority_vote(y_test, rec_test).reindex(pred_rec.index)
        score_rec = (
            pd.Series(win_score, index=rec_test)
            .groupby(level=0)
            .mean()
            .reindex(pred_rec.index)
        )
        y_true_rec = true_rec.to_numpy().astype(int)
        y_pred_rec = pred_rec.to_numpy().astype(int)
        m = heart_macc(y_true_rec, y_pred_rec, y_score_rec=score_rec.to_numpy())
        cm_png = os.path.join(figures_dir, f"cm_heart_{fs_label}_{model_name}.png")
        save_cm(
            y_true_rec,
            y_pred_rec,
            HEART_LABELS,
            f"heart {fs_label} {model_name} (recording-level)",
            cm_png,
        )
        row = {
            "modality": "heart",
            "feature_set": fs_label,
            "model": model_name,
            "primary_metric_name": "MAcc",
            "primary_metric": float(m["MAcc"]),
            "MAcc": float(m["MAcc"]),
            "Se": float(m["Se"]),
            "Sp": float(m["Sp"]),
            "macro_f1": float(m["macro_f1"]),
            "auc_roc": float(m.get("auc_roc", float("nan"))),
            "accuracy": float(m["accuracy"]),
            "n_train": volumetrics["n_train_segments"],
            "n_test": volumetrics["n_test_recordings"],
            "cm_figure": os.path.basename(cm_png),
        }
    else:
        # Cycle-level ICBHI Score (§Pattern 6). No recording aggregation.
        m = icbhi_score(y_test, preds, normal_label=LUNG_NORMAL_LABEL)
        pcs = per_class_se(y_test, preds, LUNG_LABELS)
        cm_png = os.path.join(figures_dir, f"cm_lung_{fs_label}_{model_name}.png")
        save_cm(
            y_test,
            preds,
            LUNG_LABELS,
            f"lung {fs_label} {model_name} (cycle-level)",
            cm_png,
        )
        row = {
            "modality": "lung",
            "feature_set": fs_label,
            "model": model_name,
            "primary_metric_name": "ICBHI_Score",
            "primary_metric": float(m["ICBHI_Score"]),
            "ICBHI_Score": float(m["ICBHI_Score"]),
            "Se": float(m["Se"]),
            "Sp": float(m["Sp"]),
            "macro_f1": float(macro_f1(y_test, preds)),
            "auc_roc": "",  # AUC not defined for the 4-class ICBHI headline (Pattern 8)
            "accuracy": float(accuracy(y_test, preds)),
            "se_crackle": pcs[0],
            "se_wheeze": pcs[1],
            "se_both": pcs[2],
            "se_normal": pcs[3],
            "n_train": volumetrics["n_train_segments"],
            "n_test": volumetrics["n_test_segments"],
            "cm_figure": os.path.basename(cm_png),
        }

    # Volumetric block (shared shape with classical rows — §Pattern 9) + DL extras.
    row.update(
        {
            "train_time_s": float(train_time_s),
            "n_train_segments": volumetrics["n_train_segments"],
            "n_test_segments": volumetrics["n_test_segments"],
            "n_train_recordings": volumetrics["n_train_recordings"],
            "n_test_recordings": volumetrics["n_test_recordings"],
            "n_train_patients": volumetrics["n_train_patients"],
            "n_test_patients": volumetrics["n_test_patients"],
            "params": int(params_count),  # D-09 DL volumetric field
        }
    )
    # Raw arrays for the smoke non-degenerate-CM contract (not part of the CSV schema).
    row["test_true"] = y_test
    row["test_pred"] = preds
    row["cm_figure_path"] = cm_png
    return row


# ---------------------------------------------------------------------------
# per-experiment driver: build_loaders -> model -> train_one_model -> evaluate
# ---------------------------------------------------------------------------
def run_modality(
    cache,
    modality,
    model="cnn",
    lr=None,
    batch_size=32,
    max_epochs=30,
    patience=7,
    wall_cap_s=1800,
    out_dir=None,
    device=None,
    seed=42,
    weight_decay=0.0,
    label_smoothing=0.0,
    aug_strength=1.0,
    sampler_mode="class_weight",
    cnn_widths=None,
    p=0.3,
    **_ignored,
):
    """Train + evaluate ONE DL experiment (modality × model); return its row dict.

    Builds leakage-safe loaders (``build_loaders`` re-asserts (train,test)+(train,val)
    disjointness), constructs the model (``cnn`` → ``SmallCNN`` / ``effnet_b0`` →
    ``build_efficientnet_b0``), trains with weighted CE (TRAIN-only class weights, D-05) and
    the modality-appropriate val monitor (MAcc heart / ICBHI lung), then evaluates on TEST.
    ``lr`` defaults to 1e-3 for the small CNN and 1e-4 for EfficientNet (the caller may
    override). Writes a learning-curve PNG + checkpoint under ``out_dir`` (PNG/ckpt only — no
    CSV). Returns the Phase-3-shaped row dict (with ``params``, ``epochs_ran``,
    ``curve_png``, and raw ``test_true``/``test_pred``).

    ``seed`` controls val-carve split + DataLoader shuffle + model weight init. Set all
    global RNGs (random, numpy, torch) here so multi-seed callers only need to pass the int.

    HPO knobs (all default to 04-04 values so no-arg call is behaviour-identical to 04-04):
      - ``weight_decay``: L2 regularisation for Adam (default 0.0 — 04-04 behaviour).
      - ``label_smoothing``: CrossEntropyLoss label smoothing (default 0.0 — 04-04 behaviour).
      - ``aug_strength``: SpecAugment mask-param + noise scaling for the TRAIN set only
        (default 1.0 — 04-04 behaviour; threaded into build_loaders).
      - ``sampler_mode``: imbalance strategy — ``"class_weight"`` (default, weighted CE) or
        ``"weighted_sampler"`` (WeightedRandomSampler + unweighted CE, never double-applied).
      - ``cnn_widths``: 4-tuple of SmallCNN channel widths (default None → (16,32,64,128)).
      - ``p``: SmallCNN dropout ≥ 0.3 (default 0.3 — 04-04 behaviour).
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_name = "effnet_b0" if str(model).lower() in ("effnet", "effnet_b0", "efficientnet") else "cnn"
    for_effnet = model_name == "effnet_b0"
    if lr is None:
        lr = 1e-4 if for_effnet else 1e-3

    out_dir = out_dir or os.path.join(FIGURES_DIR, f"{modality}_{model_name}")
    os.makedirs(out_dir, exist_ok=True)
    figures_dir = out_dir
    fs_label = "log_mel_64x128"

    # Build loaders (leakage re-assert lives inside build_loaders → assert_no_patient_leakage).
    # Thread aug_strength + sampler_mode as HPO knobs; defaults reproduce 04-04 exactly.
    loaders = build_loaders(
        cache, modality, for_effnet=for_effnet, batch_size=batch_size, seed=seed,
        aug_strength=aug_strength, sampler_mode=sampler_mode,
    )
    n_classes = loaders["n_classes"]

    # Redundant explicit leakage guard at the driver top (defence in depth, D-03 / T-04-07):
    # build_loaders already asserts, but a second check here documents the contract at the
    # single train+evaluate entry point the Plan-04 driver calls.
    split = np.asarray(cache["split"])
    pid = np.asarray(cache["patient_id"])
    assert_no_patient_leakage(pid[split == "train"], pid[split == "test"])

    # Volumetrics (segment/recording/patient counts — §Pattern 9).
    rec_id = np.asarray(cache["recording_id"])
    is_tr, is_te = split == "train", split == "test"
    volumetrics = {
        "n_train_segments": int(is_tr.sum()),
        "n_test_segments": int(is_te.sum()),
        "n_train_recordings": len(set(map(str, rec_id[is_tr]))),
        "n_test_recordings": len(set(map(str, rec_id[is_te]))),
        "n_train_patients": len(set(map(str, pid[is_tr]))),
        "n_test_patients": len(set(map(str, pid[is_te]))),
    }

    # Model.
    if for_effnet:
        # CPU head-only freeze fallback (D-04) when no GPU is available (deadline protection).
        freeze = not torch.cuda.is_available()
        net = build_efficientnet_b0(n_classes, freeze_backbone=freeze)
    else:
        # Thread HPO width + dropout knobs; defaults reproduce 04-04 exactly.
        net = SmallCNN(
            n_classes=n_classes,
            p=p,
            widths=tuple(cnn_widths) if cnn_widths is not None else (16, 32, 64, 128),
        )
    params_count = count_params(net)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Imbalance strategy: when using WeightedRandomSampler the sampler handles class
    # imbalance — do NOT also apply class-weighted CE loss (T-04-13: never double-apply).
    if sampler_mode == "weighted_sampler":
        # Unweighted CE + sampler (two strategies are mutually exclusive, D-05).
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        # Default "class_weight" path: weighted CE + standard shuffle (04-04 behaviour).
        criterion = nn.CrossEntropyLoss(
            weight=loaders["class_weights"].to(dev),
            label_smoothing=label_smoothing,
        )

    val_metric_fn = _val_macc if modality == "heart" else _val_icbhi

    curve_png = os.path.join(out_dir, f"learning_curve_{modality}_{model_name}.png")
    ckpt_path = os.path.join(out_dir, f"ckpt_{modality}_{model_name}.pt")

    net, train_info = train_one_model(
        net,
        loaders["train_loader"],
        loaders["val_loader"],
        criterion,
        lr=lr,
        val_metric_fn=val_metric_fn,
        max_epochs=max_epochs,
        patience=patience,
        wall_cap_s=wall_cap_s,
        ckpt_path=ckpt_path,
        curve_png=curve_png,
        device=dev,
        weight_decay=weight_decay,
    )

    row = evaluate(
        net,
        loaders["test_loader"],
        modality,
        loaders["test_recording_id"],
        figures_dir,
        fs_label,
        model_name,
        params_count,
        train_info["train_time_s"],
        volumetrics,
        device=dev,
    )
    # Attach training diagnostics + artifact paths for the caller / smoke contracts.
    row["best_val_score"] = train_info["best_val_score"]
    row["epochs_ran"] = train_info["epochs_ran"]
    row["lr"] = float(lr)
    row["curve_png"] = curve_png
    row["learning_curve_png"] = curve_png
    row["ckpt_path"] = ckpt_path
    return row
