"""Parameter initialization using basic tensor operations."""
import math

import torch


@torch.no_grad()
def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                      a: float = -3.0, b: float = 3.0) -> torch.Tensor:
    """Sample a conditioned normal by inverse CDF, then copy into tensor.

    This is truncation, not clipping unbounded normal samples. Compute the
    inverse CDF in at least float32 for low-precision parameter tensors.
    """
    if not tensor.is_floating_point():
        raise TypeError("initialization requires a real floating-point tensor")
    if not all(math.isfinite(v) for v in (mean, std, a, b)) or std <= 0 or a >= b:
        raise ValueError("require finite mean/bounds, std > 0 and a < b")
    lower = math.erf((a - mean) / (std * math.sqrt(2)))
    upper = math.erf((b - mean) / (std * math.sqrt(2)))
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    values = torch.empty(tensor.shape, dtype=dtype, device=tensor.device)
    values.uniform_(lower, upper)
    # Guard inverse-erf endpoints against floating-point rounding.
    limit = 1 - torch.finfo(dtype).eps
    values.clamp_(-limit, limit).erfinv_()
    values.mul_(std * math.sqrt(2)).add_(mean).clamp_(a, b)
    tensor.copy_(values)
    return tensor


def truncated_xavier_normal_(tensor: torch.Tensor) -> torch.Tensor:
    """Xavier normal for a matrix, truncated at +/-3 standard deviations."""
    if tensor.ndim != 2 or any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError("Xavier initialization requires a nonempty matrix")
    fan_out, fan_in = tensor.shape
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return truncated_normal_(tensor, std=std, a=-3 * std, b=3 * std)
