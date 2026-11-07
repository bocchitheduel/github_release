"""Save and restore complete training checkpoints."""
from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Any, BinaryIO, Callable

import numpy as np
import torch


def save_checkpoint(path: str | os.PathLike | BinaryIO, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, step: int, *,
                    extra: dict[str, Any] | None = None, scheduler=None,
                    scaler=None, data_rng: np.random.Generator | None = None) -> None:
    """Save training state to a path or an open binary file object.

    Paths use atomic replacement. File objects are written at their current
    position and remain open; flushing and seeking are the caller's responsibility.

    Pass loader.rng as data_rng to resume the memmap loader's sampling stream.
    step should be the number of completed optimizer updates. extra may hold
    model configuration, epoch, metrics, and learning-rate schedule settings.
    """
    if not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    state = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "extra": extra if extra is not None else {},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "data": data_rng.bit_generator.state if data_rng is not None else None,
        },
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if not isinstance(path, (str, os.PathLike)):
        torch.save(state, path)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,
                                         prefix=f".{destination.name}.", delete=False) as stream:
            temporary_path = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path: str | os.PathLike | BinaryIO, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None, *,
                    map_location="cpu", strict: bool = True, restore_rng: bool = True,
                    scheduler=None, scaler=None,
                    data_rng: np.random.Generator | None = None,
                    validate_extra: Callable[[dict[str, Any]], None] | None = None) -> int:
    """Restore existing components and return the integer iteration count.

    Construct the model/optimizer on the desired device before loading;
    map_location controls checkpoint storage loading, not model placement.
    Full RNG checkpoints contain Python/NumPy objects, so only load trusted
    files (torch.load uses weights_only=False).
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format version")
    if validate_extra is not None:
        validate_extra(state["extra"])
    # Validate optional resume components before modifying model weights.
    for component, name in [(scheduler, "scheduler"), (scaler, "scaler")]:
        if component is not None and state[name] is None:
            raise ValueError(f"checkpoint has no {name} state")
    rng = state["rng"]
    if restore_rng and data_rng is not None:
        if rng["data"] is None:
            raise ValueError("checkpoint has no data_rng state")
        if rng["data"]["bit_generator"] != type(data_rng.bit_generator).__name__:
            raise ValueError("data_rng bit generator does not match checkpoint")
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and torch.cuda.is_available():
            if len(rng["cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count does not match saved RNG states")
            torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        if data_rng is not None:
            data_rng.bit_generator.state = rng["data"]
    return int(state["step"])
