"""
src/ast_model.py — Audio Spectrogram Transformer wrapper (Phase 4, MODL-02 / ADV-02).

Provides ``build_ast(n_classes, checkpoint=...)`` and ``_to_ast_input(spec)`` adapter so
the AST fine-tune path in ``scripts/run_ast.py`` is a thin caller of the SAME
``src.train_cnn.train_one_model + evaluate`` loop as the CNN / EfficientNet experiments,
preserving byte-comparability in the unified results table.

  - ``_to_ast_input(spec)`` — per-clip min-max scale (leakage-free, mirrors
    ``src.datasets._to_effnet_image``), then adapt the project's ``(64, 128)`` log-mel dB
    cache to the AST checkpoint's ``(num_mel_bins, max_length)`` contract via bilinear
    interpolation along the frequency axis and zero-padding / trimming along the time axis.
    Returns a 1-D ``input_values`` tensor of shape ``(num_mel_bins * max_length,)`` as
    expected by ``ASTForAudioClassification.forward(input_values=...)``.

  - ``build_ast(n_classes, checkpoint=...)`` — lazy-import ``transformers`` (so this module
    is importable on a machine without transformers; the import error is raised only when
    ``build_ast`` is actually called). Downloads / loads the AudioSet AST checkpoint via
    ``AutoModelForAudioClassification.from_pretrained``, replaces the AudioSet 527-class
    head with an ``n_classes`` head (``ignore_mismatched_sizes=True``), and wraps the whole
    model in ``ASTWrapper`` so its ``forward(x)`` accepts a ``(B, 1, 64, 128)`` batch and
    returns ``(B, n_classes)`` logits — exactly the same forward contract as ``SmallCNN``
    and EfficientNet-B0.

  - ``ASTWrapper`` — thin ``nn.Module`` that applies ``_to_ast_input`` per-sample, stacks
    them into the ``(B, num_mel_bins * max_length)`` ``input_values`` tensor, and calls the
    underlying ``ASTForAudioClassification`` forward, returning the logits tensor.

``count_params`` is RE-EXPORTED from ``src.cnn`` (single source of truth — never
re-implemented here).

``import config`` runs first for the SEED=42 determinism side effect.
"""
import config  # noqa: F401 — import FIRST for the SEED=42 side effect (determinism)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cnn import count_params  # noqa: F401 — re-export; single source of truth (D-09)

__all__ = ["build_ast", "_to_ast_input", "ASTWrapper", "count_params"]

# Default pretrained AudioSet AST checkpoint (public HF Hub, no token needed).
_DEFAULT_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"

# AST checkpoint's expected mel/time spec dimensions:
# num_mel_bins=128, max_length=1024 — from the ASTConfig of this checkpoint.
# These are read off the loaded config at build time (not hard-coded blindly), but we
# also expose them as module-level constants so _to_ast_input can be called without
# building the full model (e.g., for dataset collation on the GPU box).
_AST_NUM_MEL_BINS = 128
_AST_MAX_LENGTH = 1024


def _to_ast_input(spec, num_mel_bins=_AST_NUM_MEL_BINS, max_length=_AST_MAX_LENGTH):
    """Adapt a ``(64, 128)`` dB log-mel tensor to AST's ``input_values`` 1-D vector.

    Steps (all per-clip, leakage-free — mirrors ``src.datasets._to_effnet_image``):

    1. Min-max scale to [0, 1] using only THIS clip's min/max (no dataset statistics).
    2. Frequency axis: bilinearly interpolate 64 mel bins → ``num_mel_bins`` (128 for
       the AudioSet checkpoint).
    3. Time axis: trim or zero-pad the 128 time frames → ``max_length`` (1024 for the
       AudioSet checkpoint).
    4. Flatten (freq, time) → 1-D ``input_values`` of shape ``(num_mel_bins * max_length,)``
       as expected by ``ASTForAudioClassification.forward(input_values=...)``.

    Parameters
    ----------
    spec : torch.Tensor, shape (64, 128) or (1, 64, 128)
        Log-mel dB spectrogram from the project's ``(64, 128)`` cache.
    num_mel_bins : int
        Target frequency bins (read from the checkpoint's ASTConfig; default 128).
    max_length : int
        Target time frames (read from the checkpoint's ASTConfig; default 1024).

    Returns
    -------
    torch.Tensor, shape ``(num_mel_bins * max_length,)``
        Float32 ``input_values`` vector ready for ``ASTForAudioClassification.forward``.
    """
    # Accept (1, 64, 128) from the DataLoader (SpectrogramDataset returns (1,H,W)) or
    # bare (64, 128) when called directly.
    if spec.dim() == 3:
        spec = spec.squeeze(0)  # (64, 128)

    # 1. Per-clip min-max scaling to [0, 1] (leakage-free — only this clip's stats).
    s = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)  # (64, 128)

    # 2. Frequency interpolation: 64 → num_mel_bins via bilinear resize.
    #    Treat the (freq, time) spec as a (1, 1, H, W) image for F.interpolate.
    s_4d = s.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 128)
    s_resized = F.interpolate(
        s_4d, size=(num_mel_bins, 128), mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)  # (num_mel_bins, 128)

    # 3. Time axis: pad to max_length (zero-pad on the right) or trim.
    T = s_resized.shape[1]  # currently 128
    if T < max_length:
        pad_cols = max_length - T
        s_time = F.pad(s_resized, (0, pad_cols))  # (num_mel_bins, max_length)
    else:
        s_time = s_resized[:, :max_length]  # trim to max_length

    # 4. Flatten to the 1-D input_values expected by ASTForAudioClassification.
    return s_time.reshape(-1)  # (num_mel_bins * max_length,)


class ASTWrapper(nn.Module):
    """Wrap ``ASTForAudioClassification`` to accept ``(B, 1, 64, 128)`` batches.

    Applies ``_to_ast_input`` per-sample, stacks the results into the
    ``(B, num_mel_bins * max_length)`` ``input_values`` tensor, and delegates to the
    underlying HuggingFace ``ASTForAudioClassification`` forward, returning the logits
    tensor ``(B, n_classes)`` — the SAME forward contract as ``SmallCNN`` / EffNet so
    ``src.train_cnn.train_one_model + evaluate`` can drive it unchanged.
    """

    def __init__(self, hf_model, num_mel_bins=_AST_NUM_MEL_BINS, max_length=_AST_MAX_LENGTH):
        super().__init__()
        self.hf_model = hf_model
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length

    def forward(self, x):
        """Forward pass: ``x`` shape ``(B, 1, 64, 128)`` → logits ``(B, n_classes)``.

        Applies ``_to_ast_input`` per sample and calls the HuggingFace model.
        Returns the raw logits tensor (no softmax — consistent with SmallCNN / EffNet).
        """
        # Adapt each sample in the batch independently (leakage-free per-clip scaling).
        adapted = torch.stack(
            [_to_ast_input(x[i], self.num_mel_bins, self.max_length) for i in range(x.shape[0])]
        )  # (B, num_mel_bins * max_length)
        out = self.hf_model(input_values=adapted)
        return out.logits  # (B, n_classes)


def build_ast(n_classes, checkpoint=_DEFAULT_CHECKPOINT):
    """Build an ``ASTWrapper`` fine-tunable from the AudioSet AST checkpoint.

    Lazy-imports ``transformers`` so this module is importable on a machine without
    transformers installed (the ImportError is raised here, not at module import time,
    so local test collection skips cleanly).

    Downloads (or loads from HF cache) ``checkpoint`` via
    ``AutoModelForAudioClassification.from_pretrained`` with
    ``ignore_mismatched_sizes=True`` so the AudioSet 527-class head is replaced by a
    fresh ``n_classes`` head. The returned ``ASTWrapper`` accepts ``(B, 1, 64, 128)``
    batches and returns ``(B, n_classes)`` logits.

    Parameters
    ----------
    n_classes : int
        Number of output classes (2 for heart binary, 4 for lung ICBHI).
    checkpoint : str
        HuggingFace Hub model ID (default ``"MIT/ast-finetuned-audioset-10-10-0.4593"``).

    Returns
    -------
    ASTWrapper
        Wraps the HuggingFace ``ASTForAudioClassification`` with the ``(B,1,64,128)``
        forward contract.
    """
    try:
        from transformers import AutoModelForAudioClassification, ASTConfig
    except ImportError as e:
        raise ImportError(
            "The `transformers` library is required for build_ast but is not installed. "
            "On the GPU box: `pip install transformers==5.9.0`. "
            "Local test collection skips this path automatically."
        ) from e

    hf_model = AutoModelForAudioClassification.from_pretrained(
        checkpoint,
        num_labels=n_classes,
        ignore_mismatched_sizes=True,  # replaces the 527-class AudioSet head
    )

    # Read the actual mel/time spec dimensions from the loaded checkpoint config so
    # the adapter is always consistent with the pretrained weights (not blindly hardcoded).
    cfg = hf_model.config
    num_mel_bins = getattr(cfg, "num_mel_bins", _AST_NUM_MEL_BINS)
    max_length = getattr(cfg, "max_length", _AST_MAX_LENGTH)

    return ASTWrapper(hf_model, num_mel_bins=num_mel_bins, max_length=max_length)
