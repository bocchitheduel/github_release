"""Bias-free linear transformation with truncated Xavier normal weights."""
import torch
from torch import nn
from parameter_init import truncated_xavier_normal_


class Linear(nn.Module):
    """Compute y = x W^T for inputs shaped (..., in_features).

    W has shape (out_features, in_features). Initialization uses Xavier
    normal std = sqrt(2 / (in_features + out_features)), truncated to +/-3 std.
    """

    def __init__(self, in_features: int, out_features: int,
                 device=None, dtype=None):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features,
                                          device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        truncated_xavier_normal_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.transpose(0, 1)
