"""
Audio Spectrogram Transformer wrapper.

Wraps a pretrained AudioSet AST checkpoint so it exposes the same
``forward(x) -> logits`` contract as ``SmallCNN`` / EfficientNet-B0, letting the
shared training and evaluation loop drive it unchanged. ``count_params`` is
re-exported from ``src.cnn``.
"""
from src import config

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cnn import count_params

__all__ = ["build_ast", "_to_ast_input", "ASTWrapper", "count_params"]

_DEFAULT_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"

_AST_NUM_MEL_BINS = 128
_AST_MAX_LENGTH = 1024


def _to_ast_input(spec, num_mel_bins=_AST_NUM_MEL_BINS, max_length=_AST_MAX_LENGTH):
    """Adapt a ``(64, 128)`` dB log-mel tensor to AST's ``input_values``.

    All steps are per-clip and leakage-free: min-max scale using only this clip's
    stats, bilinearly resize the frequency axis to ``num_mel_bins``, then pad or
    trim the time axis to ``max_length``.

    Parameters
    ----------
    spec : torch.Tensor, shape (64, 128) or (1, 64, 128)
        Log-mel dB spectrogram from the project's cache.

    Returns
    -------
    torch.Tensor, shape ``(max_length, num_mel_bins)``
        ``input_values`` tensor for ``ASTForAudioClassification.forward``.
    """
    if spec.dim() == 3:
        spec = spec.squeeze(0)

    s = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

    s_4d = s.unsqueeze(0).unsqueeze(0)
    s_resized = F.interpolate(
        s_4d, size=(num_mel_bins, 128), mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)

    T = s_resized.shape[1]
    if T < max_length:
        pad_cols = max_length - T
        s_time = F.pad(s_resized, (0, pad_cols))
    else:
        s_time = s_resized[:, :max_length]

    return s_time.transpose(0, 1).contiguous()


class ASTWrapper(nn.Module):
    """Wrap ``ASTForAudioClassification`` to accept ``(B, 1, 64, 128)`` batches.

    Adapts each sample via ``_to_ast_input`` and returns ``(B, n_classes)`` logits,
    matching the ``SmallCNN`` / EfficientNet forward contract.
    """

    def __init__(self, hf_model, num_mel_bins=_AST_NUM_MEL_BINS, max_length=_AST_MAX_LENGTH):
        super().__init__()
        self.hf_model = hf_model
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length

    def forward(self, x):
        """``(B, 1, 64, 128)`` -> raw logits ``(B, n_classes)`` (no softmax)."""
        adapted = torch.stack(
            [_to_ast_input(x[i], self.num_mel_bins, self.max_length) for i in range(x.shape[0])]
        )
        out = self.hf_model(input_values=adapted)
        return out.logits


def build_ast(n_classes, checkpoint=_DEFAULT_CHECKPOINT):
    """Build a fine-tunable ``ASTWrapper`` from the AudioSet AST checkpoint.

    ``transformers`` is imported lazily so this module stays importable without it.
    Loads ``checkpoint`` with ``ignore_mismatched_sizes=True`` so the AudioSet
    527-class head is replaced by a fresh ``n_classes`` head.

    Parameters
    ----------
    n_classes : int
        Number of output classes (2 for heart, 4 for lung).
    checkpoint : str
        HuggingFace Hub model ID.

    Returns
    -------
    ASTWrapper
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
        ignore_mismatched_sizes=True,
    )

    cfg = hf_model.config
    num_mel_bins = getattr(cfg, "num_mel_bins", _AST_NUM_MEL_BINS)
    max_length = getattr(cfg, "max_length", _AST_MAX_LENGTH)

    return ASTWrapper(hf_model, num_mel_bins=num_mel_bins, max_length=max_length)
