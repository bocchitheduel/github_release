"""Token embeddings with vocabulary rows: y = x W."""
import torch
from torch import nn
from parameter_init import truncated_normal_


class Embedding(nn.Module):
    """Map token IDs of any shape to (*token_ids.shape, embedding_dim).

    W has shape (vocab_size, embedding_dim). With row-oriented one-hot inputs
    x of shape (..., vocab_size), y = x @ W.
    Token IDs select rows of W without constructing the one-hot tensor.
    Initialization samples N(0, 1), conditioned on values within [-3, 3].
    ID range checks are opt-in with validate_ids=True; normal training assumes
    token IDs have already been validated during data preparation.
    """

    def __init__(self, vocab_size: int = 784, embedding_dim: int = 256,
                 device=None, dtype=None, *, validate_ids: bool = False):
        super().__init__()
        if vocab_size <= 0 or embedding_dim <= 0:
            raise ValueError("vocab_size and embedding_dim must be positive")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.validate_ids = validate_ids
        self.W = nn.Parameter(torch.empty(vocab_size, embedding_dim,
                                          device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        truncated_normal_(self.W, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute y = x W by indexing with the token IDs represented by x."""
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must have dtype torch.long")
        if token_ids.device != self.W.device:
            raise ValueError("token_ids and embedding weights must be on the same device")
        # PyTorch indexing accepts negative indices, but token IDs must remain
        # in [0, vocab_size). Autograd accumulates repeated lookups by row.
        if self.validate_ids and torch.any((token_ids < 0) | (token_ids >= self.vocab_size)):
            raise IndexError("token ID is outside the vocabulary")
        return self.W[token_ids]


TokenEmbedding = Embedding
