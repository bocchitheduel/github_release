"""RMS normalization with float32 computation."""
import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize the last dimension and apply a trainable per-feature gain.

    y = x / sqrt(mean(x^2) + eps) * weight. Computation, including the gain,
    uses float32, and the output is cast back to the input dtype.
    """

    def __init__(self, d_model: int = 784, eps: float = 1e-5,
                 device=None, dtype=None):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("RMSNorm input must have a floating-point dtype")
        if x.ndim == 0 or x.shape[-1] != self.d_model:
            raise ValueError(f"input's last dimension must be {self.d_model}")
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        output = x_float * inverse_rms * self.weight.to(torch.float32)
        return output.to(input_dtype)
