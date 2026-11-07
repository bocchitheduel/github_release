"""Numerically stable cross-entropy computed directly from logits."""
import torch
from torch import nn


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                  reduction: str = "mean", *, validate_ids: bool = False) -> torch.Tensor:
    """Compute next-token cross-entropy with classes on the final axis.

    logits: (..., vocab_size); targets: (...) of dtype torch.long.
    Subtract the largest logit before exponentiation. Low-precision logits
    are promoted to float32 for loss computation; gradients remain connected.
    ID range checks default off; use validate_ids=True for debugging. Training
    inputs should come from validated binary files produced by prepare_data.
    """
    if not logits.is_floating_point():
        raise TypeError("logits must have a floating-point dtype")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a nonempty final vocabulary dimension")
    if targets.dtype != torch.long:
        raise TypeError("targets must have dtype torch.long")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match logits.shape[:-1]")
    if targets.device != logits.device:
        raise ValueError("logits and targets must be on the same device")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be none, mean, or sum")
    if validate_ids and torch.any((targets < 0) | (targets >= logits.shape[-1])):
        raise IndexError("target token ID is outside the vocabulary")
    scores = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    # This shift cancels algebraically from the loss. Detaching it avoids
    # backpropagating through amax without changing the logits gradient.
    shifted = scores - scores.detach().amax(dim=-1, keepdim=True)
    log_partition = shifted.exp().sum(dim=-1).log()
    target_scores = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    losses = log_partition - target_scores
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


class CrossEntropyLoss(nn.Module):
    """Module wrapper for stable cross-entropy with vocabulary on the last axis."""

    def __init__(self, reduction: str = "mean", *, validate_ids: bool = False):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be none, mean, or sum")
        self.reduction = reduction
        self.validate_ids = validate_ids

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits, targets, self.reduction, validate_ids=self.validate_ids)
