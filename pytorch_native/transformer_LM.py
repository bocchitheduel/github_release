"""Four-layer causal Transformer language model."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from embedding_module import Embedding
from linear_module import Linear
from rms_normalization import RMSNorm
from transformer_block import TransformerBlock


@torch.compiler.disable
def _apply_fused_linear_cross_entropy(loss_function, weight, hidden, targets):
    """Run Liger's custom autograd op outside Inductor's traced graph.

    Liger 0.8.5 calls ``Tensor.item()`` in its forward and currently fails
    Inductor lowering if Dynamo captures its internal matmuls. Keeping the
    call as a graph break still lets ``Module.compile()`` compile the model
    trunk and sends the custom loss op through its supported eager path.
    """
    return loss_function(weight, hidden, targets)


class TransformerLM(nn.Module):
    """Embedding -> pre-norm blocks with RoPE -> final norm -> linear.

    Forward returns logits of shape (batch, sequence, vocab) unless targets are
    supplied with the optional fused training-loss path.
    forward_logits provides the same output; forward_last_logits projects only
    the final position and returns (batch, vocab). Each block applies RoPE to
    its queries and keys; embeddings are not rotated separately.
    """

    def __init__(self, vocab_size: int = 784, d_model: int = 512,
                 num_layers: int = 4, num_heads: int = 16, d_ffn: int = 1344,
                 max_seq_len: int = 512, eps: float = 1e-5,
                 theta: float = 10000.0, device=None, dtype=None, *,
                 q_block_size: int = 64, kv_block_size: int = 64,
                 attention_backend: str = "sdpa",
                 checkpoint_group_size: int = 2,
                 fused_lm_head_ce: bool = False):
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        if (not isinstance(checkpoint_group_size, int)
                or isinstance(checkpoint_group_size, bool)
                or checkpoint_group_size < 0):
            raise ValueError("checkpoint_group_size must be a non-negative integer")
        self.checkpoint_group_size = checkpoint_group_size
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
        self._fused_lm_head_ce = None
        if fused_lm_head_ce:
            try:
                from liger_kernel.transformers.fused_linear_cross_entropy import (
                    LigerFusedLinearCrossEntropyLoss,
                )
            except ImportError as error:
                raise RuntimeError(
                    "fused_lm_head_ce requires the optional liger-kernel dependency; "
                    "install requirements-fused-lm-head-ce.txt"
                ) from error
            self._fused_lm_head_ce = LigerFusedLinearCrossEntropyLoss(
                reduction="mean", accum_dtype=torch.float32)

    def _forward_hidden(self, token_ids: torch.Tensor,
                        token_positions: torch.Tensor | None = None, *,
                        offset: int = 0) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        x = self.embedding(token_ids)
        group_size = self.checkpoint_group_size
        if group_size == 0 or not self.training or not torch.is_grad_enabled():
            for block in self.blocks:
                x = block(x, token_positions, offset=offset)
        else:
            for group_start in range(0, self.num_layers, group_size):
                group_end = min(group_start + group_size, self.num_layers)

                def run_group(hidden, start=group_start, end=group_end):
                    for block_index in range(start, end):
                        hidden = self.blocks[block_index](
                            hidden, token_positions, offset=offset)
                    return hidden

                x = activation_checkpoint(run_group, x, use_reentrant=False)
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
                offset: int = 0,
                targets: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits, or a scalar fused loss when targets are supplied."""
        if targets is not None:
            return self.forward_loss(token_ids, targets, token_positions,
                                     offset=offset)
        return self.forward_logits(token_ids, token_positions, offset=offset)

    def forward_loss(self, token_ids: torch.Tensor, targets: torch.Tensor,
                     token_positions: torch.Tensor | None = None, *,
                     offset: int = 0) -> torch.Tensor:
        """Compute training loss without materializing full vocabulary logits.

        This path is available only when ``fused_lm_head_ce=True``. The normal
        logits path remains unchanged for generation and callers that need
        per-position scores.
        """
        if self._fused_lm_head_ce is None:
            raise RuntimeError("construct TransformerLM with fused_lm_head_ce=True")
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if targets.shape != token_ids.shape or targets.dtype != torch.long:
            raise ValueError("targets must be torch.long and match token_ids.shape")
        if targets.device != token_ids.device:
            raise ValueError("token_ids and targets must be on the same device")
        hidden = self.final_norm(
            self._forward_hidden(token_ids, token_positions, offset=offset))
        hidden = hidden.reshape(-1, self.d_model)
        targets = targets.reshape(-1)
        return _apply_fused_linear_cross_entropy(
            self._fused_lm_head_ce, self.lm_head.W, hidden, targets)
