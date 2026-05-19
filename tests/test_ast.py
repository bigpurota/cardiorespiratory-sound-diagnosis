"""Tests for the AST (Audio Spectrogram Transformer) model path.

Skip-on-missing: imports happen inside test bodies so collection never errors on a
machine without ``transformers`` — those tests skip instead.

Covers:
  - ``test_build_ast_forward_shape`` — ``build_ast(n_classes)`` returns a model whose
    forward maps ``(B, 1, 64, 128)`` → ``(B, n_classes)`` via ``_to_ast_input``.
    Skipped if ``transformers`` or ``src.ast_model`` is absent (no HF checkpoint).
  - ``test_ast_input_adapter_shape`` — ``_to_ast_input`` on a synthetic ``(64, 128)``
    tensor returns the expected 2-D shape. Does not require ``transformers``.
  - ``test_unified_adds_ast`` — once AST rows are present, ``unified_comparison.csv``
    contains ``model=='ast'`` rows and preserves the prior classical + cnn/effnet rows.
"""
import importlib
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity


def _import(module_name):
    """Import ``module_name``, skipping (not erroring) if it is absent."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover — defensive (module not yet present)
        pytest.skip(f"{module_name} not implemented yet or missing dependency: {exc}")


def _require_transformers():
    """Skip the test if ``transformers`` is not installed."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed — skipping AST tests")


# ---------------------------------------------------------------------------
# _to_ast_input adapter shape (no transformers needed)
# ---------------------------------------------------------------------------

def test_ast_input_adapter_shape():
    """``_to_ast_input((64,128))`` returns a 2-D ``(max_length, num_mel_bins)`` tensor.

    AST's ``input_values`` contract is ``(batch, max_length, num_mel_bins)``, so the
    per-clip adapter must emit ``(max_length, num_mel_bins) = (1024, 128)``, not a
    flattened 1-D vector (which triggers a "Dimension out of range" error in the AST
    patch embedding). This test does not require ``transformers``.
    """
    ast_mod = _import("src.ast_model")
    if not hasattr(ast_mod, "_to_ast_input"):
        pytest.skip("src.ast_model._to_ast_input not implemented yet")

    import torch  # noqa: E402

    # Default AST checkpoint dimensions: num_mel_bins=128, max_length=1024.
    num_mel_bins = 128
    max_length = 1024

    spec = torch.randn(64, 128)
    out = ast_mod._to_ast_input(spec, num_mel_bins=num_mel_bins, max_length=max_length)
    expected_shape = (max_length, num_mel_bins)  # (1024, 128)
    assert out.dim() == 2, f"_to_ast_input must return a 2-D tensor; got shape {tuple(out.shape)}"
    assert tuple(out.shape) == expected_shape, (
        f"_to_ast_input must return shape {expected_shape}; got {tuple(out.shape)}"
    )

    # Also accept (1, 64, 128) input (DataLoader adds the channel dim).
    spec_chan = torch.randn(1, 64, 128)
    out_chan = ast_mod._to_ast_input(spec_chan, num_mel_bins=num_mel_bins, max_length=max_length)
    assert tuple(out_chan.shape) == expected_shape, (
        f"_to_ast_input must handle (1,64,128) input; got {tuple(out_chan.shape)}"
    )


# ---------------------------------------------------------------------------
# build_ast forward shape: (B,1,64,128) -> (B,n_classes)  [needs transformers]
# ---------------------------------------------------------------------------

def test_build_ast_forward_shape():
    """``build_ast(n_classes).forward((B,1,64,128))`` → ``(B,n_classes)``.

    Skipped if ``transformers`` is not installed or ``src.ast_model`` is absent. This
    downloads the AST checkpoint on first run, so it only runs where transformers and
    network access are available.
    """
    _require_transformers()
    ast_mod = _import("src.ast_model")
    if not hasattr(ast_mod, "build_ast"):
        pytest.skip("src.ast_model.build_ast not implemented yet")

    import torch  # noqa: E402

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

    # 4-class head (lung modality).
    n_lung = 4
    model4 = ast_mod.build_ast(n_lung)
    model4.eval()
    with torch.no_grad():
        out4 = model4(batch)
    assert tuple(out4.shape) == (B, n_lung), (
        f"build_ast(4) forward must emit (B,4); got {tuple(out4.shape)}"
    )


# ---------------------------------------------------------------------------
# unified_comparison.csv gains model='ast' rows, prior rows preserved
# ---------------------------------------------------------------------------

def test_unified_adds_ast():
    """``unified_comparison.csv`` gains ``model=='ast'`` rows once the AST run lands.

    Skipped if no AST row is present yet. When AST rows are present, also asserts that
    the 16 classical + 4 cnn/effnet rows (20 prior rows) are still intact so the
    idempotent merge has not clobbered earlier results.
    """
    import pathlib  # noqa: E402
    import sys  # noqa: E402

    # pandas may or may not be available; skip cleanly if absent.
    try:
        import pandas as pd  # noqa: E402
    except ImportError:
        pytest.skip("pandas not installed — skipping unified_adds_ast check.")

    csv_path = PROJECT_ROOT / "results" / "tables" / "unified_comparison.csv"
    if not csv_path.exists():
        pytest.skip(f"unified_comparison.csv not found at {csv_path}; AST run not yet done.")

    df = pd.read_csv(csv_path)

    # No AST rows yet means the AST run hasn't happened — skip gracefully.
    n_ast = int((df["model"] == "ast").sum())
    if n_ast == 0:
        pytest.skip(
            "No model='ast' rows in unified_comparison.csv yet. "
            "Re-run this test after scripts/run_ast.py completes."
        )

    # At least one AST row exists: verify the prior 20 rows are still present.
    n_prior = int((df["model"] != "ast").sum())
    assert n_prior >= 20, (
        f"Expected >= 20 non-AST rows (16 classical + 4 cnn/effnet) to be preserved; "
        f"got {n_prior}. The merge may have clobbered prior rows."
    )

    # Expect one AST row per modality once the full run completed.
    assert n_ast in (1, 2), (
        f"Expected 1 or 2 model='ast' rows (1 per modality); got {n_ast}."
    )

    # Total row count: 20 prior + 1–2 AST.
    total = len(df)
    assert 20 <= total <= 22, (
        f"Expected 20–22 total rows in unified_comparison.csv; got {total}."
    )

    print(f"[test_unified_adds_ast] PASS — {n_ast} AST rows, {n_prior} prior rows, "
          f"{total} total rows.")
