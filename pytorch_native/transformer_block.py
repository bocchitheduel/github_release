"""Pre-normalized Transformer block with RMSNorm and residual connections."""
import torch
from torch import nn

from MH_selfattention import MultiHeadSelfAttention
from ffn_module import SwiGLU
from rms_normalization import RMSNorm


class TransformerBlock(nn.Module):
    """x = x + Attention(RMSNorm(x)); y = x + FFN(RMSNorm(x)).

    Inputs and outputs are (batch, sequence, d_model). The SwiGLU FFN applies
    independently at each position; attention uses a causal mask and RoPE.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 16,
                 d_ffn: int = 1344, max_seq_len: int = 512,
                 eps: float = 1e-5, theta: float = 10000.0,
                 device=None, dtype=None, *, q_block_size: int = 64,
                 kv_block_size: int = 64,
                 attention_backend: str = "sdpa"):
        super().__init__()
        self.d_model = d_model
        self.attn_norm = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.attention = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype,
            q_block_size=q_block_size, kv_block_size=kv_block_size,
            attention_backend=attention_backend)
        self.ffn_norm = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ffn, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None,
                *, offset: int = 0) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x), token_positions, offset=offset)
        return x + self.ffn(self.ffn_norm(x))
