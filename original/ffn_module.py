"""Bias-free SwiGLU feed-forward network."""
import torch
from torch import nn
from parameter_init import truncated_xavier_normal_


class SwiGLU(nn.Module):
    """FFN(x) = (SiLU(x W_gate^T) * (x W_up^T)) W_down^T.

    All weights use Xavier normal initialization truncated at +/-3 sigma,
    where sigma = sqrt(2 / (fan_in + fan_out)). SiLU(x) = x * sigmoid(x).
    Inputs and outputs have shape (..., d_model).
    """

    def __init__(self, d_model: int = 512, d_ffn: int = 1344,
                 device=None, dtype=None):
        super().__init__()
        if d_model <= 0 or d_ffn <= 0:
            raise ValueError("d_model and d_ffn must be positive")
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.W_gate = nn.Parameter(torch.empty(d_ffn, d_model, device=device, dtype=dtype))
        self.W_up = nn.Parameter(torch.empty(d_ffn, d_model, device=device, dtype=dtype))
        self.W_down = nn.Parameter(torch.empty(d_model, d_ffn, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.W_gate, self.W_up, self.W_down):
            truncated_xavier_normal_(weight)

    @staticmethod
    def silu(x: torch.Tensor) -> torch.Tensor:
        """Compute SiLU directly using a numerically stable sigmoid."""
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(x @ self.W_gate.transpose(0, 1))
        up = x @ self.W_up.transpose(0, 1)
        return (gate * up) @ self.W_down.transpose(0, 1)


FFN = SwiGLU
