"""Causal multi-head self-attention with adjacent-pair RoPE."""
import torch
from torch import nn
from torch.nn import functional as F

from linear_module import Linear
from RoPE_module import RoPE
from block_attention import block_causal_attention


class MultiHeadSelfAttention(nn.Module):
    """Map (batch, sequence, d_model) to the same shape.

    Causality follows sequence order, independently of supplied RoPE positions.
    All four projections are bias-free and use truncated Xavier initialization.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 16,
                 max_seq_len: int = 512, theta: float = 10000.0,
                 device=None, dtype=None, *, q_block_size: int = 64,
                 kv_block_size: int = 64,
                 attention_backend: str = "sdpa"):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        if q_block_size <= 0 or kv_block_size <= 0:
            raise ValueError("attention block sizes must be positive")
        if attention_backend not in ("block", "sdpa"):
            raise ValueError("attention_backend must be 'block' or 'sdpa'")
        self.q_block_size = q_block_size
        self.kv_block_size = kv_block_size
        self.attention_backend = attention_backend
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RoPE(self.head_dim, max_seq_len, theta, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,
                *, offset: int = 0, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply attention with an optional mask broadcastable to (B, H, L, L).

        Boolean True allows attention; floating masks add to attention scores.
        The causal mask still applies. For per-batch padding use (B, 1, 1, L).
        """
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"input must have shape (batch, sequence, {self.d_model})")
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        def split_heads(projection):
            return projection(x).reshape(batch_size, seq_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)

        q = self.rope(split_heads(self.q_proj), token_positions, offset=offset)
        k = self.rope(split_heads(self.k_proj), token_positions, offset=offset)
        v = split_heads(self.v_proj)
        if self.attention_backend == "block":
            attended = block_causal_attention(
                q, k, v, self.q_block_size, self.kv_block_size,
                attn_mask=attn_mask)
        else:
            attended = self._sdpa(q, k, v, attn_mask)
        combined = attended.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(combined)

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              attn_mask: torch.Tensor | None) -> torch.Tensor:
        """Run PyTorch SDPA with the same upper-left causal mask semantics."""
        if attn_mask is None:
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0)

        if attn_mask.dtype != torch.bool and not attn_mask.is_floating_point():
            raise TypeError("attn_mask must be boolean or floating-point")
        if attn_mask.device != q.device:
            raise ValueError("attn_mask must be on the same device as Q")
        expected = (q.shape[0], self.num_heads, q.shape[-2], k.shape[-2])
        try:
            mask = attn_mask.expand(expected)
        except RuntimeError as error:
            raise ValueError("attn_mask must broadcast to (batch, heads, query, key)") from error

        causal = torch.ones((q.shape[-2], k.shape[-2]), dtype=torch.bool,
                            device=q.device).tril()
        if mask.dtype == torch.bool:
            mask = mask & causal
        else:
            mask = mask.to(dtype=q.dtype).masked_fill(~causal, -torch.inf)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0)


MH_SelfAttention = MultiHeadSelfAttention
