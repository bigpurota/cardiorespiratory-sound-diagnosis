"""Deep-learning training and evaluation for the heart/lung"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import copy

from src import config

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import (
    majority_vote,
    heart_macc,
    icbhi_score,
    per_class_se,
    macro_f1,
    accuracy,
    save_cm,
)
from src.split import assert_no_patient_leakage
from src.datasets import build_loaders
from src.cnn import SmallCNN, build_efficientnet_b0, count_params

__all__ = ["train_one_model", "evaluate", "run_modality"]

HEART_LABELS = [0, 1]
LUNG_LABELS = [0, 1, 2, 3]
LUNG_NORMAL_LABEL = 3

RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def _val_macc(y_true, y_pred):
    """Heart early-stop monitor: window-level MAcc."""
    return float(heart_macc(np.asarray(y_true), np.asarray(y_pred))["MAcc"])


def _val_icbhi(y_true, y_pred):
    """Lung early-stop monitor: cycle-level ICBHI score."""
    return float(
        icbhi_score(np.asarray(y_true), np.asarray(y_pred), normal_label=LUNG_NORMAL_LABEL)[
            "ICBHI_Score"
        ]
    )


def _recalibrate_batchnorm(model, train_loader, device, n_passes=3, max_batches=None):
    """Recompute BatchNorm running mean/var as an exact average"""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    trainable_bn = [
        m for m in bns
        if getattr(m, "weight", None) is not None and m.weight.requires_grad
    ]
    if not trainable_bn:
        return
    saved_momentum = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    was_training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(max(1, n_passes)):
            for i, (xb, _yb) in enumerate(train_loader):
                if max_batches is not None and i >= max_batches:
                    break
                model(xb.to(device))
    for m, mom in zip(bns, saved_momentum):
        m.momentum = mom
    if not was_training:
        model.eval()


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
    """Train ``model`` with early stop and a wall-clock cap,"""
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
            if time.perf_counter() - t0 > wall_cap_s:
                capped_mid_epoch = True
                break
        tr_losses.append(run / max(1, seen))

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
            break
        if capped_mid_epoch or time.perf_counter() - t0 > wall_cap_s:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

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
        fig.tight_layout()
        fig.savefig(curve_png, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return model, {
        "best_val_score": float(best_score),
        "train_time_s": float(time.perf_counter() - t0),
        "epochs_ran": len(tr_losses),
    }


def _predict_test(model, test_loader, device):
    """Run test inference; return (y_true, y_pred,"""
    model.eval()
    y_true, y_pred, score1 = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            prob = torch.softmax(out, dim=1).cpu().numpy()
            y_pred.append(out.argmax(1).cpu().numpy())
            y_true.append(np.asarray(yb))
            score1.append(prob[:, 1] if prob.shape[1] > 1 else prob[:, 0])
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(score1),
    )


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
    """Evaluate ``model`` on the test set and return the metric +"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(figures_dir, exist_ok=True)

    y_test, preds, win_score = _predict_test(model, test_loader, device)

    if modality == "heart":
        rec_test = np.asarray(list(map(str, rec_test)), dtype=object)
        pred_rec = majority_vote(preds, rec_test)
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
            "auc_roc": "",
            "accuracy": float(accuracy(y_test, preds)),
            "se_crackle": pcs[0],
            "se_wheeze": pcs[1],
            "se_both": pcs[2],
            "se_normal": pcs[3],
            "n_train": volumetrics["n_train_segments"],
            "n_test": volumetrics["n_test_segments"],
            "cm_figure": os.path.basename(cm_png),
        }

    row.update(
        {
            "train_time_s": float(train_time_s),
            "n_train_segments": volumetrics["n_train_segments"],
            "n_test_segments": volumetrics["n_test_segments"],
            "n_train_recordings": volumetrics["n_train_recordings"],
            "n_test_recordings": volumetrics["n_test_recordings"],
            "n_train_patients": volumetrics["n_train_patients"],
            "n_test_patients": volumetrics["n_test_patients"],
            "params": int(params_count),
        }
    )
    row["test_true"] = y_test
    row["test_pred"] = preds
    row["cm_figure_path"] = cm_png
    return row


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
    """Train and evaluate one experiment (modality x model);"""
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

    loaders = build_loaders(
        cache, modality, for_effnet=for_effnet, batch_size=batch_size, seed=seed,
        aug_strength=aug_strength, sampler_mode=sampler_mode,
    )
    n_classes = loaders["n_classes"]

    split = np.asarray(cache["split"])
    pid = np.asarray(cache["patient_id"])
    assert_no_patient_leakage(pid[split == "train"], pid[split == "test"])

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

    if for_effnet:
        freeze = not torch.cuda.is_available()
        net = build_efficientnet_b0(n_classes, freeze_backbone=freeze)
    else:
        net = SmallCNN(
            n_classes=n_classes,
            p=p,
            widths=tuple(cnn_widths) if cnn_widths is not None else (16, 32, 64, 128),
        )
    params_count = count_params(net)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if sampler_mode == "weighted_sampler":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
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
    row["best_val_score"] = train_info["best_val_score"]
    row["epochs_ran"] = train_info["epochs_ran"]
    row["lr"] = float(lr)
    row["curve_png"] = curve_png
    row["learning_curve_png"] = curve_png
    row["ckpt_path"] = ckpt_path
    return row
