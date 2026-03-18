"""
src/cnn.py — deep-learning model factory (Phase 4, MODL-02).

The model half of the Phase-4 DL comparative study. Pure ``nn.Module`` definitions and
builders only — there is NO training logic here (that lives in ``src/train_cnn.py``):

  - ``SmallCNN(n_classes, p=0.3, widths=(16,32,64,128))`` — the from-scratch baseline (D-06):
    four Conv2d-BatchNorm2d-ReLU-MaxPool2d blocks (channels 1→widths[0]→…→widths[3]), where
    the default ``widths=(16,32,64,128)`` reproduces the 04-04 architecture exactly.
    AdaptiveAvgPool2d, then Flatten → Dropout(p≥0.3) → Linear(widths[-1], n_classes). Accepts
    ``(B, 1, 64, 128)`` log-mel batches and emits ``(B, n_classes)`` logits (04-RESEARCH §Code
    Examples 4). HPO knobs: ``widths`` and ``p``.
  - ``build_efficientnet_b0(n_classes, freeze_backbone=False)`` — the transfer-learning
    path (D-04): ``timm.create_model("efficientnet_b0", pretrained=True, in_chans=3,
    num_classes=n_classes)`` (VERIFIED exactly 4,010,110 params at num_classes=2). The
    ``(3, 224, 224)`` ImageNet-normalised image is produced by the Dataset
    (``src.datasets._to_effnet_image``), NOT here. ``freeze_backbone=True`` is the CPU
    head-only fallback: freeze every parameter, then re-enable only the classifier head.
  - ``count_params(model)`` — total parameter count (D-09 volumetric field).

``import config`` runs first for the SEED=42 determinism side effect.
"""
import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import torch.nn as nn
import timm

# Re-export the canonical EfficientNet image adapter (single source of truth lives in
# src.datasets — NOT re-implemented here) so the (1,64,128)→(3,224,224) lift the timm
# backbone expects is reachable from src.cnn for the forward-shape contract (D-04).
from src.datasets import _to_effnet_image  # noqa: F401 — re-export, not re-implementation

__all__ = ["SmallCNN", "build_efficientnet_b0", "count_params", "_to_effnet_image"]


class SmallCNN(nn.Module):
    """From-scratch 4-conv-block CNN: ``(B, 1, 64, 128)`` → ``(B, n_classes)`` (D-06).

    Four Conv2d(3, padding=1)-BatchNorm2d-ReLU-MaxPool2d(2) blocks grow the channels
    as specified by ``widths`` (default ``(16, 32, 64, 128)`` — the 04-04 architecture),
    AdaptiveAvgPool2d(1) collapses the spatial dims, and the head applies
    Dropout(``p``, default 0.3 ≥ 0.3 per D-06) before the final Linear(widths[-1], n_classes).
    ``forward`` returns raw logits (a weighted ``CrossEntropyLoss`` is applied downstream).

    HPO knobs (additive — defaults reproduce the 04-04 architecture exactly):
      - ``widths``: 4-tuple of channel widths for the 4 conv blocks (D-06 default 16/32/64/128).
        Changing widths changes the model capacity (and param count) without touching architecture.
      - ``p``: dropout probability (≥ 0.3 per D-06 floor).
    """

    def __init__(self, n_classes, p=0.3, widths=(16, 32, 64, 128)):
        super().__init__()
        if p < 0.3:
            # D-06: regularise the from-scratch baseline with dropout >= 0.3.
            raise ValueError(f"SmallCNN dropout p must be >= 0.3 (D-06); got {p}")
        widths = tuple(widths)
        if len(widths) != 4:
            raise ValueError(f"SmallCNN widths must be a 4-tuple; got {widths}")

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, kernel_size=3, padding=1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w0, w1, w2, w3 = widths
        self.features = nn.Sequential(
            block(1, w0), block(w0, w1), block(w1, w2), block(w2, w3)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),  # Dropout(p ≥ 0.3) — D-06 floor
            nn.Linear(w3, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def build_efficientnet_b0(n_classes, freeze_backbone=False):
    """Build the timm EfficientNet-B0 transfer model (4,010,110 params at n_classes=2).

    ``timm.create_model("efficientnet_b0", pretrained=True, in_chans=3,
    num_classes=n_classes)`` — ImageNet weights are fetched over HTTPS on first call and
    cached thereafter (T-04-05, accept). The ``(3, 224, 224)`` ImageNet-normalised input is
    produced upstream by ``src.datasets._to_effnet_image`` (D-04); this builder returns the
    bare backbone only.

    ``freeze_backbone=True`` is the CPU head-only fallback (D-04): every parameter is
    frozen (``requires_grad=False``) and then only the classifier head is re-enabled, so a
    CPU run trains a few-thousand-parameter head instead of the full 4M-param network.
    """
    m = timm.create_model(
        "efficientnet_b0", pretrained=True, in_chans=3, num_classes=n_classes
    )
    if freeze_backbone:
        for p_ in m.parameters():
            p_.requires_grad = False
        for p_ in m.get_classifier().parameters():
            p_.requires_grad = True
    return m


def count_params(model):
    """Total parameter count of ``model`` (D-09 volumetric field)."""
    return int(sum(p.numel() for p in model.parameters()))
