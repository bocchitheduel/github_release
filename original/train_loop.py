"""Train a Transformer LM on raw binary token-ID memmaps."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch

from amp_utils import bf16_autocast, validate_bf16_amp
from checkpoint import load_checkpoint, save_checkpoint
from crossentropy import cross_entropy
from dataloader import DataLoader
from data_metadata import validate_data_metadata, validate_resume_data
from muon import DEFAULT_NS_COEFFICIENTS, Muon, OptimizerBundle, create_optimizers
from train_utils import clip_grad_norm_, cosine_learning_rate, set_cosine_learning_rate
from transformer_LM import TransformerLM


@dataclass
class TrainConfig:
    train_path: str
    val_path: str
    vocab_size: int
    output_dir: str = "checkpoints"
    resume: str | None = None
    token_dtype: str = "int64"
    device: str = "auto"
    seed: int = 42
    batch_size: int = 16
    sequence_length: int = 512
    gradient_accumulation_steps: int = 1
    d_model: int = 512
    d_ffn: int = 1344
    num_layers: int = 4
    checkpoint_group_size: int = 2
    num_heads: int = 16
    max_seq_len: int = 512
    total_steps: int = 10000
    lr_schedule_steps: int | None = None
    warmup_steps: int = 500
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    muon_lr: float | None = None
    muon_weight_decay: float | None = None
    adamw_lr: float | None = None
    adamw_weight_decay: float | None = None
    beta1: float = 0.9
    beta2: float = 0.999
    optimizer: str = "adamw"
    loss_backend: str = "torch"
    muon_momentum: float = 0.95
    muon_nesterov: bool = False
    muon_ns_steps: int = 5
    muon_ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS
    muon_ns_precision: str = "tf32x3"
    muon_update_strategy: str = "fused"
    muon_eps: float = 1e-7
    muon_scaling_mode: str = "rms_to_rms"
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    save_interval: int = 1000
    log_interval: int = 100
    compile: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    q_block_size: int = 64
    kv_block_size: int = 64
    amp: bool = True


_RESUME_MUTABLE_FIELDS = {
    "resume", "output_dir", "device", "total_steps", "log_interval", "eval_interval",
    "eval_batches", "save_interval", "compile", "compile_backend", "compile_mode", "amp",
}

_NEW_CONFIG_DEFAULTS = {
    # Checkpoints made before Muon support omitted these fields. They remain
    # compatible with the original AdamW defaults, but cannot silently resume
    # into a different optimizer configuration.
    "optimizer": "adamw",
    "loss_backend": "torch",
    "muon_momentum": 0.95,
    "muon_nesterov": False,
    "muon_ns_steps": 5,
    "muon_ns_coefficients": DEFAULT_NS_COEFFICIENTS,
    "muon_ns_precision": "tf32x3",
    "muon_update_strategy": "fused",
    "muon_eps": 1e-7,
    "muon_scaling_mode": "rms_to_rms",
    "gradient_accumulation_steps": 1,
    "checkpoint_group_size": 2,
    "scheduler": "cosine",
    "muon_lr": None,
    "muon_weight_decay": None,
    "adamw_lr": None,
    "adamw_weight_decay": None,
    "lr_schedule_steps": None,
}


def set_config_learning_rate(optimizer, iteration: int, config: TrainConfig) -> float:
    """Apply one continuous schedule to optimizer-specific base learning rates."""
    if config.scheduler == "cosine":
        schedule_steps = config.lr_schedule_steps or config.total_steps
        scheduled = cosine_learning_rate(iteration, config.max_lr, config.min_lr,
                                         schedule_steps, config.warmup_steps)
        factor = scheduled / config.max_lr if config.max_lr else 0.0
    elif config.scheduler == "warmup_constant":
        if config.warmup_steps < 0 or config.warmup_steps >= config.total_steps:
            raise ValueError("warmup_constant requires 0 <= warmup_steps < total_steps")
        factor = min((iteration + 1) / config.warmup_steps, 1.0) if config.warmup_steps else 1.0
    else:
        raise ValueError("scheduler must be 'cosine' or 'warmup_constant'")
    components = optimizer.optimizers if isinstance(optimizer, OptimizerBundle) else [optimizer]
    logged_lr = config.max_lr * factor
    for component in components:
        if isinstance(component, Muon):
            base_lr = config.muon_lr if config.muon_lr is not None else config.max_lr
        else:
            base_lr = config.adamw_lr if config.adamw_lr is not None else config.max_lr
        for group in component.param_groups:
            group["lr"] = base_lr * factor
        if isinstance(component, Muon):
            logged_lr = base_lr * factor
    return logged_lr


def validate_resume_config(config: TrainConfig, extra: dict) -> None:
    """Reject changes affecting training; allow placement/output/monitoring changes.

    total_steps controls run length; lr_schedule_steps controls the schedule horizon.
    Path and dtype spelling differences are normalized before comparison.
    """
    saved = extra.get("config")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint is missing TrainConfig; cannot safely resume")
    differences = []
    for name, current in asdict(config).items():
        if name in _RESUME_MUTABLE_FIELDS:
            continue
        if name not in saved:
            if name in ("q_block_size", "kv_block_size") and current == 64:
                # Older checkpoints predate tiled attention; default tile
                # sizes introduce no parameter/state layout changes.
                continue
            if name in _NEW_CONFIG_DEFAULTS and current == _NEW_CONFIG_DEFAULTS[name]:
                continue
            differences.append(f"{name}: missing from checkpoint")
            continue
        previous = saved[name]
        if name in ("train_path", "val_path"):
            previous, current = str(Path(previous).resolve()), str(Path(current).resolve())
        elif name == "token_dtype":
            previous, current = np.dtype(previous).str, np.dtype(current).str
        if previous != current:
            differences.append(f"{name}: checkpoint={previous!r}, current={current!r}")
    unknown = set(saved) - set(asdict(config))
    differences.extend(f"{name}: unknown checkpoint config field" for name in sorted(unknown))
    if differences:
        raise ValueError("incompatible resume TrainConfig:\n" + "\n".join(differences))


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader,
                  num_batches: int = 20, *, amp: bool = True,
                  loss_backend: str = "torch") -> float:
    """Estimate validation loss using reproducible windows without consuming RNG."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    validate_bf16_amp(loader.device, amp)
    import copy
    rng_state = copy.deepcopy(loader.rng.bit_generator.state)
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=loader.device)
    try:
        for _ in range(num_batches):
            inputs, targets = next(loader)
            with bf16_autocast(loader.device, amp):
                if loss_backend == "liger_fused":
                    loss = model(inputs, targets=targets)
                else:
                    loss = cross_entropy(model(inputs), targets)
            loss_sum += loss
        return (loss_sum / num_batches).item()
    finally:
        loader.rng.bit_generator.state = rng_state
        model.train(was_training)


def train(config: TrainConfig, *, on_metrics: Callable[[dict], None] | None = None) -> dict:
    """Train and optionally report interval metrics at each logging step.

    Callback train_loss is the mean since the previous log (or resume).
    train_loss_last_batch and checkpoint train_loss keep their original meaning.
    Throughput excludes evaluation, checkpoint I/O and the callback itself;
    the first interval includes any lazy compilation of the training graph.
    """
    if config.total_steps <= 0 or not 0 <= config.warmup_steps < config.total_steps:
        raise ValueError("require 0 <= warmup_steps < total_steps")
    schedule_steps = config.lr_schedule_steps or config.total_steps
    if schedule_steps < config.total_steps or schedule_steps <= config.warmup_steps:
        raise ValueError("lr_schedule_steps must be >= total_steps and > warmup_steps")
    if (not isinstance(config.gradient_accumulation_steps, int)
            or isinstance(config.gradient_accumulation_steps, bool)
            or config.gradient_accumulation_steps <= 0):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if (not isinstance(config.checkpoint_group_size, int)
            or isinstance(config.checkpoint_group_size, bool)
            or config.checkpoint_group_size < 0):
        raise ValueError("checkpoint_group_size must be a non-negative integer")
    if any(value <= 0 for value in (config.eval_interval, config.eval_batches,
                                     config.save_interval, config.log_interval)):
        raise ValueError("evaluation, saving and logging intervals must be positive")
    if config.sequence_length > config.max_seq_len:
        raise ValueError("sequence_length cannot exceed max_seq_len")
    if config.loss_backend not in ("torch", "liger_fused"):
        raise ValueError("loss_backend must be 'torch' or 'liger_fused'")
    data_metadata = validate_data_metadata(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if config.device == "auto" else torch.device(config.device)
    if config.loss_backend == "liger_fused" and device.type != "cuda":
        raise ValueError("loss_backend='liger_fused' requires a CUDA device")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_loader = DataLoader(config.train_path, config.batch_size, config.sequence_length,
                              dtype=config.token_dtype, device=device, seed=config.seed)
    val_loader = DataLoader(config.val_path, config.batch_size, config.sequence_length,
                            dtype=config.token_dtype, device=device, seed=config.seed + 1)
    model = TransformerLM(vocab_size=config.vocab_size, d_model=config.d_model,
                          num_layers=config.num_layers, num_heads=config.num_heads,
                          d_ffn=config.d_ffn, max_seq_len=config.max_seq_len, device=device,
                          q_block_size=config.q_block_size, kv_block_size=config.kv_block_size,
                          checkpoint_group_size=config.checkpoint_group_size,
                          fused_lm_head_ce=config.loss_backend == "liger_fused")
    optimizer = create_optimizers(model, config)
    step = 0
    if config.resume:
        saved_metrics = {}
        def validate_resume(extra):
            validate_resume_config(config, extra)
            validate_resume_data(extra, data_metadata)
            saved_metrics.update({"train_loss": extra.get("train_loss"),
                                  "val_loss": extra.get("val_loss")})
        step = load_checkpoint(config.resume, model, optimizer, data_rng=train_loader.rng,
                               validate_extra=validate_resume)
        if step > config.total_steps:
            raise ValueError("checkpoint step exceeds configured total_steps")
        if step == config.total_steps:
            return {"step": step, **saved_metrics, "checkpoint": str(config.resume)}
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / "latest.pt"
    train_loss = None
    val_loss = None
    model.train()
    validate_bf16_amp(device, config.amp)
    loss_function = cross_entropy
    if config.compile:
        # In-place Module.compile preserves parameter names/state_dict keys.
        model.compile(backend=config.compile_backend, mode=config.compile_mode)
        if config.loss_backend == "torch":
            loss_function = torch.compile(cross_entropy, backend=config.compile_backend,
                                          mode=config.compile_mode)
    if on_metrics is not None:
        interval_loss_sum = torch.zeros((), device=device)
        interval_steps = 0
        interval_train_seconds = 0.0
        train_started = time.perf_counter()
    for iteration in range(step, config.total_steps):
        lr = set_config_learning_rate(optimizer, iteration, config)
        optimizer.zero_grad(set_to_none=True)
        update_loss = torch.zeros((), device=device) if on_metrics is None else None
        for _ in range(config.gradient_accumulation_steps):
            inputs, targets = next(train_loader)
            with bf16_autocast(device, config.amp):
                if config.loss_backend == "liger_fused":
                    loss = model(inputs, targets=targets)
                else:
                    loss = loss_function(model(inputs), targets)
            (loss / config.gradient_accumulation_steps).backward()
            if on_metrics is None:
                update_loss.add_(loss.detach(), alpha=1 / config.gradient_accumulation_steps)
            else:
                interval_loss_sum.add_(
                    loss.detach(), alpha=1 / config.gradient_accumulation_steps)
        grad_norm = clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if on_metrics is not None:
            interval_steps += 1
        step = iteration + 1
        should_log = step % config.log_interval == 0 or step == config.total_steps
        should_evaluate = step % config.eval_interval == 0 or step == config.total_steps
        should_save = step % config.save_interval == 0 or step == config.total_steps
        # Synchronize the loss scalar only when it will be reported or saved.
        if should_log or should_save:
            train_loss = loss.detach().item()
        metrics_boundary = should_log or should_evaluate or should_save
        if on_metrics is not None and metrics_boundary:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            interval_train_seconds += time.perf_counter() - train_started
        if should_log and on_metrics is None:
            print(f"step={step} train_loss={update_loss.item():.6f} lr={lr:.6g} "
                  f"grad_norm={grad_norm.item():.4f}", flush=True)
        if should_evaluate:
            val_loss = evaluate_loss(model, val_loader, config.eval_batches,
                                     amp=config.amp, loss_backend=config.loss_backend)
            if on_metrics is None:
                print(f"step={step} val_loss={val_loss:.6f}", flush=True)
        if should_log and on_metrics is not None:
            interval_tokens = (interval_steps * config.batch_size * config.sequence_length
                               * config.gradient_accumulation_steps)
            on_metrics({
                "step": step,
                "tokens_seen": (step * config.batch_size * config.sequence_length
                                * config.gradient_accumulation_steps),
                "lr": lr,
                "window_steps": interval_steps,
                "train_loss": interval_loss_sum.item() / interval_steps,
                "train_loss_last_batch": train_loss,
                "val_loss": val_loss if should_evaluate else None,
                "grad_norm": grad_norm.item(),
                "train_seconds": interval_train_seconds,
                "tokens_per_second": interval_tokens / interval_train_seconds,
            })
            interval_loss_sum.zero_()
            interval_steps = 0
            interval_train_seconds = 0.0
        if should_save:
            save_checkpoint(latest, model, optimizer, step, data_rng=train_loader.rng,
                            extra={"config": asdict(config), "data_metadata": data_metadata, "train_loss": train_loss,
                                   "val_loss": val_loss})
        if on_metrics is not None and metrics_boundary:
            train_started = time.perf_counter()
    return {"step": step, "train_loss": train_loss, "val_loss": val_loss,
            "checkpoint": str(latest)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True, help="raw binary token file")
    parser.add_argument("--val-path", required=True, help="raw binary token file")
    parser.add_argument("--vocab-size", type=int, required=True)
    # Dataclass defaults define all optional CLI settings in one place.
    from dataclasses import fields
    for field in fields(TrainConfig):
        if field.name in ("train_path", "val_path", "vocab_size"):
            continue
        value = field.default
        if field.name == "muon_ns_coefficients":
            def parse_coefficients(text):
                try:
                    values = tuple(float(item.strip()) for item in text.split(","))
                except ValueError as error:
                    raise argparse.ArgumentTypeError("expected three comma-separated numbers") from error
                if len(values) != 3:
                    raise argparse.ArgumentTypeError("expected three comma-separated numbers")
                return values
            parser.add_argument("--muon-ns-coefficients", default=value,
                                type=parse_coefficients,
                                help="three comma-separated Newton-Schulz coefficients")
            continue
        if isinstance(value, bool):
            parser.add_argument("--" + field.name.replace("_", "-"), default=value,
                                action=argparse.BooleanOptionalAction)
            continue
        parser.add_argument("--" + field.name.replace("_", "-"), default=value,
                            type=type(value) if value is not None else str)
    try:
        train(TrainConfig(**vars(parser.parse_args())))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(f"训练启动/恢复校验失败：{error}")


if __name__ == "__main__":
    main()
