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
    accumulation uses float32. Safe FP32 CUDA gradients use batched foreach
    operations; extreme values use scaled norms to avoid underflow/overflow.
    Low-precision multiplication uses float32, so the coefficient is not cast
    to the gradient dtype. Nonfinite gradients raise before modification.
    """
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("max_norm must be finite and nonnegative")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    active_parameters = []
    grads = []
    seen = set()
    for parameter in parameters:
        if parameter.grad is not None and id(parameter) not in seen:
            seen.add(id(parameter))
            active_parameters.append(parameter)
            grads.append(parameter.grad)
    if not grads:
        return torch.tensor(0.0)
    if any(grad.is_sparse for grad in grads):
        raise RuntimeError("clip_grad_norm_ requires dense gradients")
    def stable_l2(values):
        if values.numel() == 0:
            return values.new_zeros(())
        magnitude = values.abs().amax()
        divisor = torch.where(magnitude == 0, torch.ones_like(magnitude), magnitude)
        return torch.linalg.vector_norm(values / divisor) * magnitude

    norm_device = grads[0].device

    def stable_clip():
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

    # The training path has FP32 gradients on one CUDA device. Multi-tensor
    # norms and scaling avoid launching a separate sequence of kernels for
    # every parameter. Keep the scaled norm implementation for low precision,
    # mixed-device inputs, and values outside FP32's safe sum-of-squares range.
    if (norm_device.type == "cuda"
            and all(grad.device == norm_device and grad.dtype == torch.float32
                    and grad.numel() > 0 for grad in grads)):
        norms = torch._foreach_norm(grads, 2)
        norm_values = torch.stack(norms)
        total_norm = torch.linalg.vector_norm(norm_values)

        # A nonfinite norm signals overflow or nonfinite input. Below this
        # bound, even a full tensor's worth of underflowed squares could affect
        # the result, so use the scaled implementation for that rare case.
        element_count = sum(grad.numel() for grad in grads)
        finfo = torch.finfo(torch.float32)
        safe_min = math.sqrt(element_count * finfo.tiny / finfo.eps)
        needs_stable_norm = ~torch.isfinite(total_norm) | (total_norm < safe_min)
        if needs_stable_norm:
            return stable_clip()

        scale = (max_norm / (total_norm + eps)).clamp(max=1.0)
        torch._foreach_mul_(grads, scale)
        return total_norm

    return stable_clip()
