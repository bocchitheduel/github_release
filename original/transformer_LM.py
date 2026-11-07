"""Four-layer causal Transformer language model."""
import torch
from torch import nn

from embedding_module import Embedding
from linear_module import Linear
from rms_normalization import RMSNorm
from transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """Embedding -> pre-norm blocks with RoPE -> final norm -> linear.

    forward returns next-token logits of shape (batch, sequence, vocab).
    forward_logits provides the same output; forward_last_logits projects only
    the final position and returns (batch, vocab). Each block applies RoPE to
    its queries and keys; embeddings are not rotated separately.
    """

    def __init__(self, vocab_size: int = 784, d_model: int = 512,
                 num_layers: int = 4, num_heads: int = 16, d_ffn: int = 1344,
                 max_seq_len: int = 512, eps: float = 1e-5,
                 theta: float = 10000.0, device=None, dtype=None, *,
                 q_block_size: int = 64, kv_block_size: int = 64,
                 attention_backend: str = "block"):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ffn=d_ffn,
                             max_seq_len=max_seq_len, eps=eps, theta=theta,
                             device=device, dtype=dtype, q_block_size=q_block_size,
                             kv_block_size=kv_block_size,
                             attention_backend=attention_backend)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def _forward_hidden(self, token_ids: torch.Tensor,
                        token_positions: torch.Tensor | None = None, *,
                        offset: int = 0) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x, token_positions, offset=offset)
        return x

    def forward_logits(self, token_ids: torch.Tensor,
                       token_positions: torch.Tensor | None = None, *,
                       offset: int = 0) -> torch.Tensor:
        """Return unnormalized next-token scores for each input position."""
        return self.lm_head(self.final_norm(
            self._forward_hidden(token_ids, token_positions, offset=offset)))

    def forward_last_logits(self, token_ids: torch.Tensor,
                            token_positions: torch.Tensor | None = None, *,
                            offset: int = 0) -> torch.Tensor:
        """Return logits only for the final input position, shaped (batch, vocab)."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain at least one position")
        hidden = self._forward_hidden(token_ids, token_positions, offset=offset)
        return self.lm_head(self.final_norm(hidden[:, -1, :]))

    def forward(self, token_ids: torch.Tensor,
                token_positions: torch.Tensor | None = None, *,
                offset: int = 0) -> torch.Tensor:
        return self.forward_logits(token_ids, token_positions, offset=offset)
