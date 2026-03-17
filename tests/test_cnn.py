"""
tests/test_cnn.py — MODL-02 / SC2 contracts (Phase 4, Wave 0).

Specifies the model + param-count contract for the Wave-1 ``src/cnn.py`` module
(04-RESEARCH.md §Code Examples 4):

  - ``SmallCNN(n_classes)``: 4-conv-block CNN accepting ``(B, 1, 64, 128)`` and emitting
    ``(B, n_classes)`` (D-06 dropout >= 0.3 head).
  - ``build_efficientnet_b0(n_classes)``: timm EfficientNet-B0 transfer model — VERIFIED
    to have exactly 4,010,110 parameters.
  - ``count_params(model)``: total parameter count (D-09 volumetric field).
  - The effnet image adapter (``_to_effnet_image`` / ``for_effnet`` routing) lifts a
    ``(1, 64, 128)`` dB image to the ``(3, 224, 224)`` ImageNet-normalized input.

UNIT flavour: import ``src.cnn`` inside the test bodies, skip-on-missing so Wave-0
collection has zero errors and the bodies stay RED until Wave 1 ships the module.
"""
import importlib
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity

# VERIFIED live in project .venv (timm 1.0.27): EfficientNet-B0 total params.
EFFNET_B0_PARAMS = 4_010_110


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if absent in Wave 0."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive (Wave 0 module absent)
        pytest.skip(f"{module_name} not implemented yet (Wave 0): {exc}")


# ---------------------------------------------------------------------------
# UNIT — param counts: SmallCNN > 0; EfficientNet-B0 == 4,010,110 (VERIFIED)
# ---------------------------------------------------------------------------

def test_param_counts():
    """``count_params(SmallCNN(2)) > 0`` and ``count_params(build_efficientnet_b0(2)) == 4_010_110``.

    The EfficientNet-B0 total parameter count is a fixed, verified invariant (4,010,110)
    that pins the exact backbone (timm ``efficientnet_b0``, in_chans=3, num_classes=2).
    """
    cnn = _import("src.cnn")
    for fn in ("count_params", "SmallCNN", "build_efficientnet_b0"):
        if not hasattr(cnn, fn):
            pytest.skip(f"src.cnn.{fn} not implemented yet (Wave 0)")

    small = cnn.SmallCNN(n_classes=2)
    assert cnn.count_params(small) > 0, "SmallCNN must have trainable parameters"

    effnet = cnn.build_efficientnet_b0(2)
    assert cnn.count_params(effnet) == EFFNET_B0_PARAMS, (
        f"EfficientNet-B0 must have exactly {EFFNET_B0_PARAMS} params "
        f"(timm efficientnet_b0, num_classes=2); got {cnn.count_params(effnet)}"
    )


# ---------------------------------------------------------------------------
# UNIT — forward shapes: (B,1,64,128) -> (B,n_classes) for both models
# ---------------------------------------------------------------------------

def test_forward_shape():
    """SmallCNN maps (B,1,64,128) -> (B,2); the effnet path maps the same batch -> (B,n_classes).

    The EfficientNet path adapts the single-channel (1,64,128) dB image to a
    (3,224,224) ImageNet input (1->3 channel repeat + bilinear resize) before the
    backbone, but the externally observed mapping is still (B,1,64,128) -> (B,n_classes).
    """
    cnn = _import("src.cnn")
    if not hasattr(cnn, "SmallCNN"):
        pytest.skip("src.cnn.SmallCNN not implemented yet (Wave 0)")

    import torch

    B = 4
    batch = torch.randn(B, 1, 64, 128)

    small = cnn.SmallCNN(n_classes=2)
    small.eval()
    with torch.no_grad():
        out = small(batch)
    assert tuple(out.shape) == (B, 2), f"SmallCNN forward must emit (B,2); got {tuple(out.shape)}"

    # EfficientNet path: route the (B,1,64,128) batch through the image adapter, then
    # the timm backbone -> (B, n_classes).
    if not hasattr(cnn, "build_efficientnet_b0"):
        pytest.skip("src.cnn.build_efficientnet_b0 not implemented yet (Wave 0)")

    adapter = (
        getattr(cnn, "_to_effnet_image", None)
        or getattr(cnn, "to_effnet_image", None)
        or getattr(cnn, "for_effnet", None)
    )
    if adapter is None:
        pytest.skip("src.cnn effnet image adapter not implemented yet (Wave 0)")

    n_classes = 2
    effnet = cnn.build_efficientnet_b0(n_classes)
    effnet.eval()
    # Adapt each (1,64,128) sample -> (3,224,224), then stack into the effnet batch.
    imgs = torch.stack([adapter(batch[i].squeeze(0)) for i in range(B)])
    assert tuple(imgs.shape) == (B, 3, 224, 224), (
        f"effnet image adapter must emit (B,3,224,224); got {tuple(imgs.shape)}"
    )
    with torch.no_grad():
        eout = effnet(imgs)
    assert tuple(eout.shape) == (B, n_classes), (
        f"EfficientNet forward must emit (B,n_classes); got {tuple(eout.shape)}"
    )
