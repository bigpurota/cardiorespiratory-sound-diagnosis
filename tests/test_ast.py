"""Tests for the AST (Audio Spectrogram Transformer) model"""
import importlib
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))


def _import(module_name):
    """Import ``module_name``, skipping (not erroring) if it is"""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        pytest.skip(f"{module_name} not implemented yet or missing dependency: {exc}")


def _require_transformers():
    """Skip the test if ``transformers`` is not installed."""
    try:
        import transformers
    except ImportError:
        pytest.skip("transformers not installed — skipping AST tests")


def test_ast_input_adapter_shape():
    """``_to_ast_input((64,128))`` returns a 2-D ``(max_length,"""
    ast_mod = _import("src.ast_model")
    if not hasattr(ast_mod, "_to_ast_input"):
        pytest.skip("src.ast_model._to_ast_input not implemented yet")

    import torch

    num_mel_bins = 128
    max_length = 1024

    spec = torch.randn(64, 128)
    out = ast_mod._to_ast_input(spec, num_mel_bins=num_mel_bins, max_length=max_length)
    expected_shape = (max_length, num_mel_bins)
    assert out.dim() == 2, f"_to_ast_input must return a 2-D tensor; got shape {tuple(out.shape)}"
    assert tuple(out.shape) == expected_shape, (
        f"_to_ast_input must return shape {expected_shape}; got {tuple(out.shape)}"
    )

    spec_chan = torch.randn(1, 64, 128)
    out_chan = ast_mod._to_ast_input(spec_chan, num_mel_bins=num_mel_bins, max_length=max_length)
    assert tuple(out_chan.shape) == expected_shape, (
        f"_to_ast_input must handle (1,64,128) input; got {tuple(out_chan.shape)}"
    )


def test_build_ast_forward_shape():
    """``build_ast(n_classes).forward((B,1,64,128))`` →"""
    _require_transformers()
    ast_mod = _import("src.ast_model")
    if not hasattr(ast_mod, "build_ast"):
        pytest.skip("src.ast_model.build_ast not implemented yet")

    import torch

    B = 2
    n_classes = 2
    batch = torch.randn(B, 1, 64, 128)

    model = ast_mod.build_ast(n_classes)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert tuple(out.shape) == (B, n_classes), (
        f"build_ast forward must emit (B,{n_classes}); got {tuple(out.shape)}"
    )

    n_lung = 4
    model4 = ast_mod.build_ast(n_lung)
    model4.eval()
    with torch.no_grad():
        out4 = model4(batch)
    assert tuple(out4.shape) == (B, n_lung), (
        f"build_ast(4) forward must emit (B,4); got {tuple(out4.shape)}"
    )


def test_unified_adds_ast():
    """``unified_comparison.csv`` gains ``model=='ast'`` rows"""
    import pathlib
    import sys

    try:
        import pandas as pd
    except ImportError:
        pytest.skip("pandas not installed — skipping unified_adds_ast check.")

    csv_path = PROJECT_ROOT / "results" / "tables" / "unified_comparison.csv"
    if not csv_path.exists():
        pytest.skip(f"unified_comparison.csv not found at {csv_path}; AST run not yet done.")

    df = pd.read_csv(csv_path)

    n_ast = int((df["model"] == "ast").sum())
    if n_ast == 0:
        pytest.skip(
            "No model='ast' rows in unified_comparison.csv yet. "
            "Re-run this test after scripts/run_ast.py completes."
        )

    n_prior = int((df["model"] != "ast").sum())
    assert n_prior >= 20, (
        f"Expected >= 20 non-AST rows (16 classical + 4 cnn/effnet) to be preserved; "
        f"got {n_prior}. The merge may have clobbered prior rows."
    )

    assert n_ast in (1, 2), (
        f"Expected 1 or 2 model='ast' rows (1 per modality); got {n_ast}."
    )

    total = len(df)
    assert 20 <= total <= 22, (
        f"Expected 20–22 total rows in unified_comparison.csv; got {total}."
    )

    print(f"[test_unified_adds_ast] PASS — {n_ast} AST rows, {n_prior} prior rows, "
          f"{total} total rows.")
