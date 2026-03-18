"""
tests/test_ast.py — Wave-0 skip-on-missing scaffold for the AST path (Phase 4, MODL-02).

Mirrors ``tests/test_cnn.py``'s importlib + pytest.skip pattern so local test collection
(on a machine without ``transformers``) never errors — tests SKIP instead.

Two contracts:

  - ``test_build_ast_forward_shape`` — ``build_ast(n_classes)`` returns a model whose
    forward maps ``(B, 1, 64, 128)`` → ``(B, n_classes)`` via ``_to_ast_input``.
    SKIPPED if ``transformers`` or ``src.ast_model`` is absent (local, no HF checkpoint).

  - ``test_ast_input_adapter_shape`` — ``_to_ast_input`` on a synthetic ``(64, 128)``
    tensor returns a 1-D vector of length ``num_mel_bins * max_length`` (128 * 1024).
    SKIPPED only if ``src.ast_model`` is absent — does NOT require ``transformers``.

  - ``test_unified_adds_ast`` — after a real GPU run, ``unified_comparison.csv`` contains
    ``model=='ast'`` rows AND preserves the 16 classical + 4 cnn/effnet rows.
    SKIPPED if the CSV lacks an ``'ast'`` row (expected until GPU run completes).

All imports are INSIDE test bodies so collection never errors on a transformers-less machine.
"""
import importlib
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # conftest import parity


def _import(module_name):
    """Import ``module_name``, skipping (not erroring) if absent in Wave 0."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover — defensive (Wave 0 module absent)
        pytest.skip(f"{module_name} not implemented yet or missing dependency: {exc}")


def _require_transformers():
    """Skip the test if ``transformers`` is not installed (GPU-box-only dependency)."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip(
            "transformers not installed locally — AST tests require the GPU box venv "
            "(transformers==5.9.0 pinned there). Skipping."
        )


# ---------------------------------------------------------------------------
# UNIT — _to_ast_input adapter shape (no transformers needed)
# ---------------------------------------------------------------------------

def test_ast_input_adapter_shape():
    """``_to_ast_input((64,128))`` returns a 1-D vector of length 128*1024=131072.

    This test does NOT require ``transformers`` — it only exercises the adapter that
    lives in ``src.ast_model`` (which is importable without transformers).
    """
    ast_mod = _import("src.ast_model")
    if not hasattr(ast_mod, "_to_ast_input"):
        pytest.skip("src.ast_model._to_ast_input not implemented yet (Wave 0)")

    import torch  # noqa: E402

    # Default AST checkpoint dimensions: num_mel_bins=128, max_length=1024.
    num_mel_bins = 128
    max_length = 1024

    spec = torch.randn(64, 128)
    out = ast_mod._to_ast_input(spec, num_mel_bins=num_mel_bins, max_length=max_length)
    expected_len = num_mel_bins * max_length  # 131072
    assert out.dim() == 1, f"_to_ast_input must return a 1-D tensor; got shape {tuple(out.shape)}"
    assert out.shape[0] == expected_len, (
        f"_to_ast_input must return length {expected_len}; got {out.shape[0]}"
    )

    # Also accept (1, 64, 128) input (DataLoader adds the channel dim).
    spec_chan = torch.randn(1, 64, 128)
    out_chan = ast_mod._to_ast_input(spec_chan, num_mel_bins=num_mel_bins, max_length=max_length)
    assert out_chan.shape[0] == expected_len, (
        f"_to_ast_input must handle (1,64,128) input; got {out_chan.shape[0]}"
    )


# ---------------------------------------------------------------------------
# UNIT — build_ast forward shape: (B,1,64,128) -> (B,n_classes)  [transformers needed]
# ---------------------------------------------------------------------------

def test_build_ast_forward_shape():
    """``build_ast(n_classes).forward((B,1,64,128))`` → ``(B,n_classes)``.

    SKIPPED if ``transformers`` is not installed (GPU-box-only dependency) or if
    ``src.ast_model`` is absent. This test DOWNLOADS the AST checkpoint on first run —
    defer to the GPU box; do not run locally unless transformers + network are available.
    """
    _require_transformers()
    ast_mod = _import("src.ast_model")
    if not hasattr(ast_mod, "build_ast"):
        pytest.skip("src.ast_model.build_ast not implemented yet (Wave 0)")

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
# INTEGRATION — unified_comparison.csv gains model='ast' rows, prior rows preserved
# ---------------------------------------------------------------------------

def test_unified_adds_ast():
    """After the GPU run, ``unified_comparison.csv`` has ``model=='ast'`` rows.

    SKIPPED if no AST row is present yet (GPU run not yet executed). When AST rows ARE
    present, also asserts that the 16 classical + 4 cnn/effnet rows (20 prior rows) are
    still intact so the idempotent merge has not clobbered earlier work.
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
        pytest.skip(f"unified_comparison.csv not found at {csv_path}; GPU run not yet done.")

    df = pd.read_csv(csv_path)

    # If no AST rows yet, the GPU run hasn't been executed — skip gracefully.
    n_ast = int((df["model"] == "ast").sum())
    if n_ast == 0:
        pytest.skip(
            "No model='ast' rows in unified_comparison.csv yet — GPU run not yet executed. "
            "Re-run this test after scripts/run_ast.py completes on the GPU box."
        )

    # At least 1 AST row exists: verify the prior 20 rows are still present.
    n_prior = int((df["model"] != "ast").sum())
    assert n_prior >= 20, (
        f"Expected >= 20 non-AST rows (16 classical + 4 cnn/effnet) to be preserved; "
        f"got {n_prior}. The idempotent merge may have clobbered prior rows."
    )

    # AST rows must have model='ast' and both modalities if the full run completed.
    assert n_ast in (1, 2), (
        f"Expected 1 or 2 model='ast' rows (1 per modality); got {n_ast}."
    )

    # Total row count must be 21 or 22 (20 prior + 1–2 AST).
    total = len(df)
    assert 20 <= total <= 22, (
        f"Expected 20–22 total rows in unified_comparison.csv; got {total}."
    )

    print(f"[test_unified_adds_ast] PASS — {n_ast} AST rows, {n_prior} prior rows, "
          f"{total} total rows.")
