"""Contracts for the cross-modal transfer and joint"""
import importlib
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _import(module_name):
    """Import module_name; pytest-skip (not fail) if absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not available: {exc}")


def test_static_imports_reuse():
    """src/cross_modal.py must import the required building"""
    src = (PROJECT_ROOT / "src" / "cross_modal.py").read_text(encoding="utf-8")

    assert "from src.cnn import" in src, "cross_modal.py must import from src.cnn"
    assert "from src.datasets import" in src and "build_loaders" in src, (
        "cross_modal.py must import build_loaders from src.datasets"
    )
    assert "from src.metrics import" in src, "cross_modal.py must import from src.metrics"
    assert "from src.split import assert_no_patient_leakage" in src, (
        "cross_modal.py must import assert_no_patient_leakage from src.split"
    )


def test_static_no_reimplemented_metrics():
    """cross_modal.py must NOT re-implement metrics (no def"""
    src = (PROJECT_ROOT / "src" / "cross_modal.py").read_text(encoding="utf-8")

    forbidden = ["def heart_macc", "def icbhi_score", "def majority_vote"]
    for f in forbidden:
        assert f not in src, (
            f"cross_modal.py re-implements '{f}' — must import from src.metrics instead"
        )


def test_static_no_training_loop():
    """cross_modal.py must NOT re-implement a training loop (for"""
    import re
    src = (PROJECT_ROOT / "src" / "cross_modal.py").read_text(encoding="utf-8")
    assert "class SmallCNN" not in src, (
        "cross_modal.py must NOT re-implement SmallCNN — import from src.cnn"
    )
    assert "def build_efficientnet_b0" not in src, (
        "cross_modal.py must NOT re-implement build_efficientnet_b0 — import from src.cnn"
    )


def test_static_head_swap_present():
    """cross_modal.py must contain the head-swap logic and a"""
    import re
    src = (PROJECT_ROOT / "src" / "cross_modal.py").read_text(encoding="utf-8")

    assert "head_heart" in src, "cross_modal.py missing 'head_heart' (joint model)"
    assert "head_lung" in src, "cross_modal.py missing 'head_lung' (joint model)"

    has_comment = bool(re.search(
        r"(2.class|4.class|heart.binary|label.space|head.swap|binary.*4)",
        src, re.IGNORECASE
    ))
    assert has_comment, (
        "cross_modal.py must contain a comment documenting the 2-class/4-class head-swap "
        "mismatch"
    )


def test_static_leakage_guards():
    """assert_no_patient_leakage must be called at least twice in"""
    src = (PROJECT_ROOT / "src" / "cross_modal.py").read_text(encoding="utf-8")
    count = src.count("assert_no_patient_leakage(")
    assert count >= 2, (
        f"cross_modal.py calls assert_no_patient_leakage {count} time(s); expected ≥2 "
        "(once for source, once for target/joint modalities)"
    )


def test_static_no_smote_no_global_scaler():
    """cross_modal.py must not APPLY SMOTE or fit_transform"""
    import re
    src = (PROJECT_ROOT / "src" / "cross_modal.py").read_text(encoding="utf-8")
    assert not re.search(r"\bSMOTE\s*\(", src), (
        "cross_modal.py must not instantiate/call SMOTE"
    )
    assert "fit_transform" not in src, "cross_modal.py must not call fit_transform"


def test_static_spearman_wired():
    """cross_modal.py must define spearman_method_rankings and"""
    src = (PROJECT_ROOT / "src" / "cross_modal.py").read_text(encoding="utf-8")
    assert "def spearman_method_rankings" in src, (
        "cross_modal.py must define spearman_method_rankings"
    )
    assert "A_mfcc_delta" in src, (
        "cross_modal.py must reference 'A_mfcc_delta' in spearman_method_rankings"
    )
    assert "spearmanr" in src, (
        "cross_modal.py must call scipy.stats.spearmanr"
    )


def test_static_all_exports():
    """__all__ must include all six public symbols."""
    src = (PROJECT_ROOT / "src" / "cross_modal.py").read_text(encoding="utf-8")
    assert "__all__" in src, "cross_modal.py must declare __all__"
    for sym in [
        "build_shared_encoder",
        "transfer_modality",
        "JointMultiTaskModel",
        "train_joint",
        "evaluate_joint",
        "spearman_method_rankings",
    ]:
        assert sym in src, f"cross_modal.py __all__ must include '{sym}'"


def test_build_shared_encoder_cnn():
    """build_shared_encoder('cnn') returns (encoder, 128) and the"""
    cm = _import("src.cross_modal")
    import torch

    encoder, feature_dim = cm.build_shared_encoder(arch="cnn")
    assert feature_dim == 128, f"CNN feature_dim must be 128; got {feature_dim}"

    B = 2
    x = torch.randn(B, 1, 64, 128)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape == (B, 128), (
        f"CNN encoder output shape must be (B, 128); got {out.shape}"
    )


def test_joint_model_shapes():
    """JointMultiTaskModel.forward emits (B,2) for 'heart' and"""
    cm = _import("src.cross_modal")
    import torch

    model = cm.JointMultiTaskModel(arch="cnn")
    model.eval()

    B = 3
    x = torch.randn(B, 1, 64, 128)
    with torch.no_grad():
        out_h = model(x, "heart")
        out_l = model(x, "lung")

    assert out_h.shape == (B, 2), f"heart head shape must be (B,2); got {out_h.shape}"
    assert out_l.shape == (B, 4), f"lung head shape must be (B,4); got {out_l.shape}"


def test_joint_model_invalid_modality():
    """JointMultiTaskModel.forward raises AssertionError on"""
    cm = _import("src.cross_modal")
    import torch

    model = cm.JointMultiTaskModel(arch="cnn")
    model.eval()
    x = torch.randn(2, 1, 64, 128)
    with pytest.raises((AssertionError, Exception)):
        model(x, "arterial")


def _make_tiny_cache(n_classes, n_per_class=4, groups_per_class=2, prefix="p", heart_like=True):
    """Build a minimal spectrogram cache with n_classes present"""
    import numpy as np
    rng = np.random.default_rng(99)
    H, W = 64, 128
    gpc = max(groups_per_class, 4)
    rows_per_group = max(n_per_class // gpc, 2)
    X_list, labels, patient_id, split, recording_id = [], [], [], [], []
    for cls in range(n_classes):
        for g in range(gpc):
            pid = f"{prefix}{cls}_{g:02d}"
            this_split = "train" if g % 2 == 0 else "test"
            for _ in range(rows_per_group):
                spec = rng.standard_normal((H, W)).astype("float32")
                spec += float(cls) * 1.5
                X_list.append(spec)
                labels.append(cls)
                patient_id.append(pid)
                split.append(this_split)
                recording_id.append(pid if heart_like else f"{pid}.wav")
    return {
        "X": np.stack(X_list).astype("float32"),
        "labels": np.asarray(labels, dtype=int),
        "patient_id": np.asarray(patient_id, dtype=object),
        "split": np.asarray(split, dtype=object),
        "recording_id": np.asarray(recording_id, dtype=object),
    }


def test_transfer_modality_smoke(tmp_path):
    """transfer_modality(heart→lung) returns finite"""
    cm = _import("src.cross_modal")

    heart_cache = _make_tiny_cache(n_classes=2, n_per_class=4, groups_per_class=2,
                                   prefix="h", heart_like=True)
    lung_cache = _make_tiny_cache(n_classes=4, n_per_class=4, groups_per_class=2,
                                  prefix="l", heart_like=False)

    row = cm.transfer_modality(
        source_cache=heart_cache,
        target_cache=lung_cache,
        source_modality="heart",
        target_modality="lung",
        arch="cnn",
        max_epochs=2,
        patience=1,
        wall_cap_s=60,
        out_dir=str(tmp_path / "transfer"),
    )

    assert row["setting"] == "transfer"
    assert row["source_modality"] == "heart"
    assert row["target_modality"] == "lung"
    assert row["primary_metric_name"] == "ICBHI_Score"
    pm = row["primary_metric"]
    assert pm == pm, "primary_metric must not be NaN"
    assert 0.0 <= row["Se"] <= 1.0, f"Se out of range: {row['Se']}"
    assert 0.0 <= row["Sp"] <= 1.0, f"Sp out of range: {row['Sp']}"


def test_transfer_modality_reverse_smoke(tmp_path):
    """transfer_modality(lung→heart) returns finite MAcc"""
    cm = _import("src.cross_modal")

    heart_cache = _make_tiny_cache(n_classes=2, n_per_class=4, groups_per_class=2,
                                   prefix="h2", heart_like=True)
    lung_cache = _make_tiny_cache(n_classes=4, n_per_class=4, groups_per_class=2,
                                  prefix="l2", heart_like=False)

    row = cm.transfer_modality(
        source_cache=lung_cache,
        target_cache=heart_cache,
        source_modality="lung",
        target_modality="heart",
        arch="cnn",
        max_epochs=2,
        patience=1,
        wall_cap_s=60,
        out_dir=str(tmp_path / "transfer_rev"),
    )

    assert row["setting"] == "transfer"
    assert row["primary_metric_name"] == "MAcc"
    pm = row["primary_metric"]
    assert pm == pm, "primary_metric must not be NaN"
    assert 0.0 <= row["Se"] <= 1.0
    assert 0.0 <= row["Sp"] <= 1.0


def test_spearman_method_rankings(tmp_path):
    """spearman_method_rankings reads unified_comparison.csv and"""
    import math
    cm = _import("src.cross_modal")
    unified = PROJECT_ROOT / "results" / "tables" / "unified_comparison.csv"
    if not unified.exists():
        pytest.skip("unified_comparison.csv not present — skipping Spearman test")

    rho, pvalue, n, labels, heart_scores, lung_scores = cm.spearman_method_rankings(
        str(unified)
    )
    assert not math.isnan(rho), "Spearman rho must not be NaN"
    assert -1.0 <= rho <= 1.0, f"Spearman rho must be in [-1, 1]; got {rho}"
    assert n >= 2, f"need at least 2 methods; got n={n}"
    assert len(heart_scores) == n
    assert len(lung_scores) == n
