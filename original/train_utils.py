"""Cosine learning-rate schedule and global L2 gradient clipping."""
import math
from collections.abc import Iterable

import torch


def cosine_learning_rate(step: int, max_lr: float, min_lr: float,
                         total_steps: int, warmup_steps: int = 0) -> float:
    """Linear warmup followed by cosine decay, with zero-based step indices.

    Warmup starts at zero and reaches max_lr at warmup_steps. Decay reaches
    min_lr at total_steps, and subsequent steps keep min_lr. Without warmup,
    step zero starts at max_lr.
    """
    if step < 0 or total_steps <= 0 or not 0 <= warmup_steps < total_steps:
        raise ValueError("require step >= 0 and 0 <= warmup_steps < total_steps")
    if not (math.isfinite(max_lr) and math.isfinite(min_lr) and 0 <= min_lr <= max_lr):
        raise ValueError("learning rates must be finite and satisfy 0 <= min_lr <= max_lr")
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def set_cosine_learning_rate(optimizer: torch.optim.Optimizer, step: int,
                             max_lr: float, min_lr: float, total_steps: int,
                             warmup_steps: int = 0) -> float:
    """Set the same scheduled learning rate on all optimizer parameter groups."""
    lr = cosine_learning_rate(step, max_lr, min_lr, total_steps, warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def clip_grad_norm_(parameters: Iterable[torch.Tensor] | torch.Tensor,
                    max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """Clip global L2 norm in place and return the original norm.

    Parameters without gradients are skipped. All gradients receive the same
    scaling factor min(1, max_norm / (norm + eps)); low-precision norm
    accumulation uses float32. Norms are scaled before squaring to avoid
    underflow/overflow. Low-precision multiplication uses float32 as well;
    the coefficient is never cast to the low-precision gradient dtype.
    Nonfinite gradients raise before modification.
    """
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("max_norm must be finite and nonnegative")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = []
    seen = set()
    for parameter in parameters:
        if parameter.grad is not None and id(parameter) not in seen:
            seen.add(id(parameter))
            grads.append(parameter.grad)
    if not grads:
        return torch.tensor(0.0)
    if any(grad.is_sparse for grad in grads):
        raise RuntimeError("clip_grad_norm_ requires dense gradients")
    norm_device = grads[0].device
    def stable_l2(values):
        if values.numel() == 0:
            return values.new_zeros(())
        magnitude = values.abs().amax()
        divisor = torch.where(magnitude == 0, torch.ones_like(magnitude), magnitude)
        return torch.linalg.vector_norm(values / divisor) * magnitude

    norms = [stable_l2(
        grad.float() if grad.dtype in (torch.float16, torch.bfloat16) else grad
    ).to(norm_device) for grad in grads]
    total_norm = stable_l2(torch.stack(norms))
    if not torch.isfinite(total_norm):
        raise RuntimeError("cannot clip nonfinite gradient norm")
    scale = (max_norm / (total_norm + eps)).clamp(max=1.0)
    for grad in grads:
        if grad.dtype in (torch.float16, torch.bfloat16):
            work = grad.float()
            work.mul_(scale.to(device=grad.device, dtype=torch.float32))
            grad.copy_(work)
        else:
            grad.mul_(scale.to(device=grad.device, dtype=grad.dtype))
    return total_norm
