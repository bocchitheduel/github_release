"""Rotary position embeddings pairing adjacent features."""
import torch
from torch import nn


class RoPE(nn.Module):
    """Rotate inputs shaped (..., seq_len, head_dim), including batches/heads.

    Pairs features (0, 1), (2, 3), etc. The sequence axis is -2.
    token_positions may be (seq_len,), (batch, seq_len) for batched multi-head
    inputs, or any shape broadcastable to x.shape[:-1]. Positions index the
    cached range [0, max_seq_len). Batch positions retain singleton head axes;
    cached sin/cos broadcast during rotation. Omitted positions use arange + offset.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512,
                 theta: float = 10000.0, device=None):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        if max_seq_len <= 0 or theta <= 0:
            raise ValueError("max_seq_len and theta must be positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = theta ** (-torch.arange(0, head_dim, 2,
                                          dtype=torch.float32, device=device) / head_dim)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,
                *, offset: int = 0) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("RoPE input must have a floating-point dtype")
        if x.ndim < 2 or x.shape[-1] != self.head_dim:
            raise ValueError(f"input must have shape (..., seq_len, {self.head_dim})")
        seq_len = x.shape[-2]
        if x.device != self.cos_cache.device:
            raise ValueError("input and RoPE buffers must be on the same device; use module.to(device)")
        if token_positions is None:
            if offset < 0 or offset + seq_len > self.max_seq_len:
                raise ValueError("positions must lie within the cached max_seq_len")
            positions = torch.arange(offset, offset + seq_len, device=x.device)
        else:
            if offset != 0:
                raise ValueError("offset cannot be combined with explicit token_positions")
            if token_positions.dtype not in (torch.int32, torch.int64):
                raise TypeError("token_positions must have dtype int32 or int64")
            positions = token_positions.to(device=x.device, dtype=torch.long)
            # Batch-specific positions are shared across intervening head axes.
            if x.ndim > 3 and positions.ndim == 2 and positions.shape == (x.shape[0], seq_len):
                positions = positions.reshape(x.shape[0], *([1] * (x.ndim - 3)), seq_len)
            try:
                broadcast_shape = torch.broadcast_shapes(positions.shape, x.shape[:-1])
            except RuntimeError as error:
                raise ValueError("token_positions must broadcast to the input token axes") from error
            if broadcast_shape != x.shape[:-1]:
                raise ValueError("token_positions must broadcast to the input token axes")
            if torch.any((positions < 0) | (positions >= self.max_seq_len)):
                raise ValueError("positions must lie within the cached max_seq_len")
        cos = self.cos_cache[positions].to(dtype=x.dtype)
        sin = self.sin_cache[positions].to(dtype=x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin,
                            odd * cos + even * sin), dim=-1).flatten(-2)


RotaryPositionalEmbedding = RoPE
