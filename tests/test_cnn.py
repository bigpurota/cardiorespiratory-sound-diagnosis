"""Model and parameter-count contracts for ``src/cnn.py``."""
import importlib
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

EFFNET_B0_PARAMS = 4_010_110


def _import(module_name):
    """Import `module_name`, skipping (not erroring) if it is"""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not implemented yet: {exc}")


def test_param_counts():
    """``count_params(SmallCNN(2)) > 0`` and"""
    cnn = _import("src.cnn")
    for fn in ("count_params", "SmallCNN", "build_efficientnet_b0"):
        if not hasattr(cnn, fn):
            pytest.skip(f"src.cnn.{fn} not implemented yet")

    small = cnn.SmallCNN(n_classes=2)
    assert cnn.count_params(small) > 0, "SmallCNN must have trainable parameters"

    effnet = cnn.build_efficientnet_b0(2)
    assert cnn.count_params(effnet) == EFFNET_B0_PARAMS, (
        f"EfficientNet-B0 must have exactly {EFFNET_B0_PARAMS} params "
        f"(timm efficientnet_b0, num_classes=2); got {cnn.count_params(effnet)}"
    )

    small_explicit = cnn.SmallCNN(n_classes=2, widths=(16, 32, 64, 128))
    assert cnn.count_params(small_explicit) == cnn.count_params(small), (
        "SmallCNN(2, widths=(16,32,64,128)) must have the same param count as SmallCNN(2)"
    )

    wider = cnn.SmallCNN(n_classes=2, widths=(32, 64, 128, 256))
    assert cnn.count_params(wider) > cnn.count_params(small), (
        "SmallCNN with widths=(32,64,128,256) must have more params than the default net"
    )


def test_forward_shape():
    """SmallCNN maps (B,1,64,128) -> (B,2); the effnet path maps"""
    cnn = _import("src.cnn")
    if not hasattr(cnn, "SmallCNN"):
        pytest.skip("src.cnn.SmallCNN not implemented yet")

    import torch

    B = 4
    batch = torch.randn(B, 1, 64, 128)

    small = cnn.SmallCNN(n_classes=2)
    small.eval()
    with torch.no_grad():
        out = small(batch)
    assert tuple(out.shape) == (B, 2), f"SmallCNN forward must emit (B,2); got {tuple(out.shape)}"

    if not hasattr(cnn, "build_efficientnet_b0"):
        pytest.skip("src.cnn.build_efficientnet_b0 not implemented yet")

    adapter = (
        getattr(cnn, "_to_effnet_image", None)
        or getattr(cnn, "to_effnet_image", None)
        or getattr(cnn, "for_effnet", None)
    )
    if adapter is None:
        pytest.skip("src.cnn effnet image adapter not implemented yet")

    n_classes = 2
    effnet = cnn.build_efficientnet_b0(n_classes)
    effnet.eval()
    imgs = torch.stack([adapter(batch[i].squeeze(0)) for i in range(B)])
    assert tuple(imgs.shape) == (B, 3, 224, 224), (
        f"effnet image adapter must emit (B,3,224,224); got {tuple(imgs.shape)}"
    )
    with torch.no_grad():
        eout = effnet(imgs)
    assert tuple(eout.shape) == (B, n_classes), (
        f"EfficientNet forward must emit (B,n_classes); got {tuple(eout.shape)}"
    )
