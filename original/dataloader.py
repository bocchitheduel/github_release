"""Random next-token windows from a disk-backed, one-dimensional token array."""
from __future__ import annotations

import os

import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size: int = 16,
              sequence_length: int = 32, device: str | torch.device = "auto", *,
              rng: np.random.Generator | None = None
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample windows with replacement; targets are shifted by one token.

    Only the sampled batch is materialized in memory. The dataset may be a
    read-only np.memmap. Returned tensors are torch.long on the chosen device.
    """
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise ValueError("dataset must be a one-dimensional integer token array")
    if len(dataset) <= sequence_length:
        raise ValueError("dataset needs at least sequence_length + 1 tokens")
    target_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if str(device) == "auto" else torch.device(device)
    rng = rng if rng is not None else np.random.default_rng()
    # Exclusive high includes the final valid start len(data) - length - 1.
    starts = rng.integers(0, len(dataset) - sequence_length, size=batch_size)
    indices = starts[:, None] + np.arange(sequence_length + 1)
    windows = np.asarray(dataset[indices], dtype=np.int64)
    if np.any(windows < 0):
        raise ValueError("sampled token IDs must be nonnegative and fit in int64")
    # Copy each slice to contiguous batch storage; never expose a writable
    # tensor view of the read-only memory-mapped corpus.
    inputs = torch.from_numpy(windows[:, :-1].copy()).to(target_device)
    targets = torch.from_numpy(windows[:, 1:].copy()).to(target_device)
    return inputs, targets


class DataLoader:
    """Infinite random-batch iterator over raw token-ID binary storage.

    The file must contain consecutive integers of ``dtype`` (default int64),
    without a header. An existing one-dimensional np.memmap is also accepted.
    Use matching dtype when writing and reading the corpus. np.memmap remains
    CPU-backed; each selected batch is transferred to ``device`` for training.
    ``seed`` gives a reproducible sampling stream local to this loader.
    """

    def __init__(self, input_path: str | os.PathLike | np.memmap,
                 batch_size: int = 16, sequence_length: int = 32, *,
                 dtype=np.int64, device: str | torch.device = "auto",
                 seed: int | None = None):
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        storage_dtype = np.dtype(dtype)
        if not np.issubdtype(storage_dtype, np.integer):
            raise ValueError("dtype must be an integer dtype")
        self.dataset = (input_path if isinstance(input_path, np.memmap)
                        else np.memmap(input_path, dtype=storage_dtype, mode="r"))
        if self.dataset.ndim != 1 or not np.issubdtype(self.dataset.dtype, np.integer):
            raise ValueError("dataset must be a one-dimensional integer token array")
        if len(self.dataset) <= sequence_length:
            raise ValueError("dataset needs at least sequence_length + 1 tokens")
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if str(device) == "auto" else torch.device(device)
        self.rng = np.random.default_rng(seed)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return get_batch(self.dataset, self.batch_size, self.sequence_length,
                         self.device, rng=self.rng)

    def __iter__(self) -> DataLoader:
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_batch()


MemmapDataLoader = DataLoader
