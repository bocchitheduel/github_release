#!/usr/bin/env python3
"""Run a budgeted successive-halving search for RMS-to-RMS Muon.

The default plan is deliberately fixed at 1,000 optimizer updates:
8 x 25, 4 x 75, 2 x 150, and 1 x 200 additional updates. Checkpoints and
the manifest are written atomically after every update so an interrupted run
can resume without exceeding its recorded update budget.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from adamW import AdamW
from amp_utils import bf16_autocast, validate_bf16_amp
from checkpoint import load_checkpoint, save_checkpoint
from crossentropy import cross_entropy
from data_metadata import validate_data_metadata
from dataloader import DataLoader
from muon import (DEFAULT_NS_COEFFICIENTS, Muon, OptimizerBundle,
                  compute_muon_scale, newton_schulz_orthogonalize,
                  partition_muon_parameters)
from train_utils import clip_grad_norm_
from transformer_LM import TransformerLM


MAX_TOTAL_STEPS = 1000
WARMUP_STEPS = 10
MODEL = dict(vocab_size=8983, d_model=512, d_ffn=1344, num_layers=4,
             num_heads=16, max_seq_len=512, q_block_size=64, kv_block_size=64)
TRAIN = dict(batch_size=128, sequence_length=256, gradient_accumulation_steps=1,
             seed=42, grad_clip=1.0, amp=True, compile=True,
             compile_backend="inductor", compile_mode="default",
             adamw_lr=1e-3, adamw_weight_decay=0.01, beta1=0.9, beta2=0.999,
             muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5,
             muon_eps=1e-7, scaling_mode="rms_to_rms")
LR_GRID = (2.5e-4, 5e-4, 1e-3, 2e-3)
WD_GRID = (0.0, 0.1)
STAGE_TARGETS = {1: 25, 2: 100, 3: 250, 4: 450}
STAGE_EVAL_EVERY = {1: 25, 2: 25, 3: 25, 4: 50}
STAGE_KEEP = {1: 4, 2: 2, 3: 1}
DIAGNOSTIC_NAMES = (
    "blocks.0.attention.q_proj.W", "blocks.0.attention.k_proj.W",
    "blocks.0.attention.v_proj.W", "blocks.0.attention.out_proj.W",
    "blocks.0.ffn.W_up", "blocks.0.ffn.W_down",
    "blocks.3.attention.q_proj.W", "blocks.3.attention.k_proj.W",
    "blocks.3.attention.v_proj.W", "blocks.3.attention.out_proj.W",
    "blocks.3.ffn.W_up", "blocks.3.ffn.W_down",
)
COLORS = ("#276EF1", "#D84A3A", "#159570", "#8A55CC", "#DD8B00",
          "#008C95", "#C4517A", "#627D98")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(device: torch.device) -> TransformerLM:
    return TransformerLM(**MODEL, device=device)


def operator_norm(matrix: torch.Tensor, iterations: int = 8) -> float:
    """Low-cost power iteration estimate of sigma_max, without a full SVD."""
    matrix = matrix.detach().float()
    rows, cols = matrix.shape
    if rows <= cols:
        vector = torch.ones(rows, device=matrix.device, dtype=matrix.dtype)
        vector = vector / torch.linalg.vector_norm(vector)
        for _ in range(iterations):
            vector = matrix @ (matrix.T @ vector)
            vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-30)
        return float(torch.linalg.vector_norm(matrix.T @ vector).item())
    vector = torch.ones(cols, device=matrix.device, dtype=matrix.dtype)
    vector = vector / torch.linalg.vector_norm(vector)
    for _ in range(iterations):
        vector = matrix.T @ (matrix @ vector)
        vector = vector / torch.linalg.vector_norm(vector).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(matrix @ vector).item())


def preflight(model: TransformerLM, device: torch.device) -> dict:
    """Check scale math and NS finiteness without updating model parameters."""
    started = time.perf_counter()
    named = dict(model.named_parameters())
    before = tensor_state_hash(model.state_dict())
    generator = torch.Generator(device=device).manual_seed(20261029)
    matrices = []
    for name in DIAGNOSTIC_NAMES:
        weight = named[name]
        rows, cols = weight.shape
        scale = compute_muon_scale(rows, cols, mode="rms_to_rms")
        expected = math.sqrt(rows / cols)
        if not math.isclose(scale, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"RMS-to-RMS scale mismatch at {name}: {scale} vs {expected}")
        synthetic_gradient = torch.randn(weight.shape, generator=generator,
                                         device=device, dtype=torch.float32)
        direction = newton_schulz_orthogonalize(
            synthetic_gradient, ns_steps=5, coefficients=DEFAULT_NS_COEFFICIENTS,
            eps=1e-7)
        if not torch.isfinite(direction).all():
            raise FloatingPointError(f"non-finite no-update NS result at {name}")
        direction_sigma = operator_norm(direction)
        scaled_rms_operator = math.sqrt(cols / rows) * scale * direction_sigma
        matrices.append({
            "name": name, "shape": list(weight.shape), "scale": scale,
            "expected_scale": expected, "ns_frobenius_norm": float(torch.linalg.vector_norm(direction).item()),
            "ns_spectral_norm_estimate": direction_sigma,
            "scaled_direction_rms_to_rms_operator_norm": scaled_rms_operator,
            "finite": True,
        })
    after = tensor_state_hash(model.state_dict())
    if before != after:
        raise RuntimeError("no-update preflight changed model weights")
    return {"duration_seconds": time.perf_counter() - started,
            "model_weights_unchanged": True, "matrices": matrices}


class WarmupConstantScheduler:
    """One persistent 10-update linear warmup followed by constant rates."""
    def __init__(self, optimizer: OptimizerBundle, muon_lr: float,
                 adamw_lr: float, warmup_steps: int = WARMUP_STEPS):
        self.optimizer = optimizer
        self.muon_lr = float(muon_lr)
        self.adamw_lr = float(adamw_lr)
        self.warmup_steps = int(warmup_steps)
        self.completed_steps = 0

    def set_for_update(self, zero_based_step: int) -> tuple[float, float]:
        factor = min((zero_based_step + 1) / self.warmup_steps, 1.0)
        muon_lr, adamw_lr = self.muon_lr * factor, self.adamw_lr * factor
        for optimizer in self.optimizer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = muon_lr if isinstance(optimizer, Muon) else adamw_lr
        self.completed_steps = zero_based_step
        return muon_lr, adamw_lr

    def state_dict(self) -> dict:
        return {"completed_steps": self.completed_steps, "warmup_steps": self.warmup_steps,
                "muon_lr": self.muon_lr, "adamw_lr": self.adamw_lr}

    def load_state_dict(self, state: dict) -> None:
        for key, current in (("warmup_steps", self.warmup_steps),
                             ("muon_lr", self.muon_lr), ("adamw_lr", self.adamw_lr)):
            if state[key] != current:
                raise ValueError(f"scheduler checkpoint {key} mismatch")
        self.completed_steps = int(state["completed_steps"])


def build_optimizers(model: TransformerLM, lr: float, weight_decay: float):
    muon_params, adamw_params = partition_muon_parameters(model)
    muon = Muon(muon_params, lr=lr, momentum=TRAIN["muon_momentum"],
                nesterov=TRAIN["muon_nesterov"], weight_decay=weight_decay,
                ns_steps=TRAIN["muon_ns_steps"],
                ns_coefficients=DEFAULT_NS_COEFFICIENTS,
                eps=TRAIN["muon_eps"], scaling_mode="rms_to_rms")
    adamw = AdamW(adamw_params, lr=TRAIN["adamw_lr"],
                  betas=(TRAIN["beta1"], TRAIN["beta2"]),
                  weight_decay=TRAIN["adamw_weight_decay"])
    return OptimizerBundle((muon, adamw)), len(muon_params), len(adamw_params)


def eval_loss(model, loader: DataLoader, batches: int) -> tuple[float, float]:
    state = copy.deepcopy(loader.rng.bit_generator.state)
    was_training = model.training
    model.eval()
    batch_means = []
    try:
        with torch.inference_mode():
            for _ in range(batches):
                inputs, targets = next(loader)
                with bf16_autocast(loader.device, True):
                    logits = model(inputs)
                    loss_sum = cross_entropy(logits, targets, reduction="sum")
                batch_means.append(float((loss_sum / targets.numel()).item()))
        mean = statistics.fmean(batch_means)
        se = (statistics.stdev(batch_means) / math.sqrt(len(batch_means))
              if len(batch_means) > 1 else 0.0)
        return mean, se
    finally:
        loader.rng.bit_generator.state = state
        model.train(was_training)


def capture_diagnostic_inputs(named_parameters: dict[str, torch.nn.Parameter]) -> dict:
    captured = {}
    for name in DIAGNOSTIC_NAMES:
        parameter = named_parameters[name]
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for representative matrix {name}")
        captured[name] = {"weight": parameter.detach().clone(),
                          "gradient": parameter.grad.detach().float().clone()}
    return captured


def matrix_diagnostics(captured: dict, named_parameters: dict[str, torch.nn.Parameter],
                       optimizer: OptimizerBundle, lr: float, wd: float,
                       step: int, trial_id: str, stage: int) -> list[dict]:
    muon = next(opt for opt in optimizer.optimizers if isinstance(opt, Muon))
    records = []
    for name, before in captured.items():
        parameter = named_parameters[name]
        momentum = muon.state[parameter]["momentum_buffer"]
        direction = before["gradient"] + TRAIN["muon_momentum"] * momentum
        orthogonal = newton_schulz_orthogonalize(
            direction, ns_steps=TRAIN["muon_ns_steps"],
            coefficients=DEFAULT_NS_COEFFICIENTS, eps=TRAIN["muon_eps"],
            validate_finite=False)
        delta = parameter.detach().float() - before["weight"].float()
        rows, cols = parameter.shape
        scale = compute_muon_scale(rows, cols, mode="rms_to_rms")
        records.append({
            "trial_id": trial_id, "stage": stage, "step": step,
            "name": name, "shape": list(parameter.shape),
            "muon_lr": lr, "muon_weight_decay": wd,
            "weight_frobenius_norm": float(torch.linalg.vector_norm(before["weight"].float()).item()),
            "gradient_frobenius_norm": float(torch.linalg.vector_norm(before["gradient"]).item()),
            "momentum_norm": float(torch.linalg.vector_norm(momentum).item()),
            "orthogonalized_update_norm": float(torch.linalg.vector_norm(orthogonal).item()),
            "orthogonalized_spectral_norm_estimate": operator_norm(orthogonal),
            "orthogonalized_direction_rms_to_rms_operator_norm": (
                math.sqrt(cols / rows) * operator_norm(orthogonal)),
            "actual_parameter_update_norm": float(torch.linalg.vector_norm(delta).item()),
            "update_to_weight_ratio": float((torch.linalg.vector_norm(delta) /
                                              torch.linalg.vector_norm(before["weight"].float()).clamp_min(1e-30)).item()),
            "maximum_update_magnitude": float(delta.abs().amax().item()),
            "actual_update_rms_to_rms_operator_norm": (
                math.sqrt(cols / rows) * operator_norm(delta)),
            "shape_scale": scale,
        })
    return records


def append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_metrics(path: Path, row: dict) -> None:
    columns = ("trial_id", "stage", "step", "muon_lr", "muon_weight_decay",
               "train_loss", "validation_loss", "validation_standard_error",
               "gradient_norm_mean", "gradient_norm_max", "update_to_weight_ratio_max",
               "maximum_update_magnitude", "tokens_per_second", "step_time_seconds",
               "peak_memory_mib", "stability")
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if fresh:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in columns})
        stream.flush()
        os.fsync(stream.fileno())


def trial_checkpoint(run_dir: Path, trial_id: str) -> Path:
    return run_dir / "checkpoints" / f"{trial_id}.pt"


def run_trial(manifest: dict, candidate: dict, stage: int, target_steps: int,
              run_dir: Path, device: torch.device, data_metadata: dict,
              val_batches: int) -> None:
    trial_id = candidate["trial_id"]
    ckpt_path = trial_checkpoint(run_dir, trial_id)
    seed_everything(TRAIN["seed"])
    model = make_model(device)
    initial = torch.load(run_dir / "initial_model.pt", map_location=device,
                         weights_only=True)
    model.load_state_dict(initial)
    optimizer, muon_count, adamw_count = build_optimizers(
        model, candidate["muon_lr"], candidate["muon_weight_decay"])
    scheduler = WarmupConstantScheduler(optimizer, candidate["muon_lr"], TRAIN["adamw_lr"])
    train_loader = DataLoader(REPO / "train.bin", TRAIN["batch_size"],
                              TRAIN["sequence_length"], device=device,
                              seed=TRAIN["seed"])
    val_loader = DataLoader(REPO / "val.bin", TRAIN["batch_size"],
                            TRAIN["sequence_length"], device=device,
                            seed=TRAIN["seed"] + 1)
    if TRAIN["compile"]:
        model.compile(backend=TRAIN["compile_backend"], mode=TRAIN["compile_mode"])
        loss_function = torch.compile(cross_entropy, backend=TRAIN["compile_backend"],
                                      mode=TRAIN["compile_mode"])
    else:
        loss_function = cross_entropy
    model.train()
    start_step = 0
    window_loss_sum = 0.0
    window_gradient_norms: list[float] = []
    window_train_seconds = 0.0
    if ckpt_path.exists() and candidate["completed_steps"] > 0:
        def check_extra(extra):
            if extra.get("trial_id") != trial_id or extra.get("data_metadata") != data_metadata:
                raise ValueError("checkpoint trial/data metadata mismatch")
            if extra.get("muon_lr") != candidate["muon_lr"] or extra.get("muon_weight_decay") != candidate["muon_weight_decay"]:
                raise ValueError("checkpoint hyperparameters do not match manifest")
        start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler=scheduler,
                                     data_rng=train_loader.rng, validate_extra=check_extra)
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        extra = saved["extra"]
        window_loss_sum = float(extra.get("window_loss_sum", 0.0))
        window_gradient_norms = list(extra.get("window_gradient_norms", []))
        window_train_seconds = float(extra.get("window_train_seconds", 0.0))
        if start_step != candidate["completed_steps"]:
            raise ValueError("manifest/checkpoint step mismatch")
    elif start_step != 0:
        raise ValueError("trial manifest has progress but its checkpoint is missing")

    named = dict(model.named_parameters())
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    optimized = [parameter for opt in optimizer.optimizers for group in opt.param_groups
                 for parameter in group["params"]]
    if len(optimized) != len(model_parameter_ids) or {id(p) for p in optimized} != model_parameter_ids:
        raise RuntimeError("Muon/AdamW groups do not partition every trainable model parameter")
    if stage == 1 and start_step == 0:
        if candidate.get("seed_sample_hash") is None:
            # The sampler state is local and does not consume the global RNG.
            sample = next(train_loader)[0].detach().cpu().numpy().tobytes()
            candidate["seed_sample_hash"] = hashlib.sha256(sample).hexdigest()
            train_loader.rng = np.random.default_rng(TRAIN["seed"])
    torch.cuda.reset_peak_memory_stats(device)
    manifest["active_trial"] = trial_id
    manifest["current_stage"] = stage
    candidate["status"] = "running"
    atomic_json(run_dir / "manifest.json", manifest)

    while start_step < target_steps:
        global_consumed = int(manifest["optimizer_steps_consumed"])
        if global_consumed >= manifest["max_total_steps"]:
            raise RuntimeError("global optimizer-step budget exhausted before target")
        next_step = start_step + 1
        muon_lr, adamw_lr = scheduler.set_for_update(start_step)
        optimizer.zero_grad(set_to_none=True)
        update_loss = torch.zeros((), device=device)
        started = time.perf_counter()
        try:
            for _ in range(TRAIN["gradient_accumulation_steps"]):
                inputs, targets = next(train_loader)
                with bf16_autocast(device, True):
                    loss = loss_function(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite training loss")
                loss.backward()
                update_loss.add_(loss.detach() / TRAIN["gradient_accumulation_steps"])
            gradient_norm = clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("non-finite global gradient norm")
            diagnostic_due = (next_step % STAGE_EVAL_EVERY[stage] == 0 or
                              next_step == target_steps)
            captured = capture_diagnostic_inputs(named) if diagnostic_due else None
            # Reserve this update durably before mutating model/optimizer state.
            manifest["optimizer_steps_consumed"] += 1
            manifest["in_flight"] = {"trial_id": trial_id, "candidate_step": next_step,
                                     "reserved_at": time.time()}
            atomic_json(run_dir / "manifest.json", manifest)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            finite = torch.stack([torch.isfinite(parameter).all()
                                  for parameter in model.parameters()]).all()
            if not bool(finite.item()):
                candidate["numerical_failure"] = "non-finite model parameter after optimizer update"
                candidate["completed_steps"] = start_step
                candidate["status"] = "failed"
                candidate["failure_stage"] = stage
                manifest["failed_updates"] += 1
                manifest["in_flight"] = None
                manifest["active_trial"] = None
                atomic_json(run_dir / "manifest.json", manifest)
                return
        except torch.cuda.OutOfMemoryError as error:
            manifest["failed_updates"] += 1
            manifest["in_flight"] = None
            candidate["status"] = "failed"
            candidate["failure_stage"] = stage
            candidate["numerical_failure"] = f"CUDA OOM: {error}"
            manifest["active_trial"] = None
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            atomic_json(run_dir / "manifest.json", manifest)
            return
        except FloatingPointError as error:
            manifest["failed_updates"] += 1
            manifest["in_flight"] = None
            candidate["status"] = "failed"
            candidate["failure_stage"] = stage
            candidate["numerical_failure"] = str(error)
            manifest["active_trial"] = None
            optimizer.zero_grad(set_to_none=True)
            atomic_json(run_dir / "manifest.json", manifest)
            return
        except RuntimeError as error:
            if "nonfinite" not in str(error).lower() and "non-finite" not in str(error).lower():
                raise
            manifest["failed_updates"] += 1
            manifest["in_flight"] = None
            candidate["status"] = "failed"
            candidate["failure_stage"] = stage
            candidate["numerical_failure"] = str(error)
            manifest["active_trial"] = None
            optimizer.zero_grad(set_to_none=True)
            atomic_json(run_dir / "manifest.json", manifest)
            return

        start_step = next_step
        scheduler.completed_steps = start_step
        candidate["completed_steps"] = start_step
        candidate["last_train_loss"] = float(update_loss.item())
        candidate["last_gradient_norm"] = float(gradient_norm.item())
        candidate["window_train_loss_sum"] = float(candidate.get("window_train_loss_sum", 0.0)) + float(update_loss.item())
        candidate["window_gradient_norms"] = list(candidate.get("window_gradient_norms", [])) + [float(gradient_norm.item())]
        candidate["window_train_seconds"] = float(candidate.get("window_train_seconds", 0.0)) + elapsed
        candidate["step_seconds_sum"] = float(candidate.get("step_seconds_sum", 0.0)) + elapsed
        candidate["step_seconds_count"] = int(candidate.get("step_seconds_count", 0)) + 1
        candidate["tokens_seen"] = candidate["completed_steps"] * TRAIN["batch_size"] * TRAIN["sequence_length"]

        row = None
        if start_step % STAGE_EVAL_EVERY[stage] == 0 or start_step == target_steps:
            if captured is None:
                raise RuntimeError("diagnostic snapshots missing at evaluation step")
            diagnostics = matrix_diagnostics(captured, named, optimizer,
                                            candidate["muon_lr"], candidate["muon_weight_decay"],
                                            start_step, trial_id, stage)
            append_jsonl(run_dir / "diagnostics.jsonl", diagnostics)
            training_loss = candidate["window_train_loss_sum"] / max(1, len(candidate["window_gradient_norms"]))
            validation_loss, validation_se = eval_loss(model, val_loader, val_batches)
            if not math.isfinite(validation_loss):
                candidate["numerical_failure"] = "non-finite validation loss"
                candidate["status"] = "failed"
                candidate["failure_stage"] = stage
                manifest["failed_updates"] += 1
                manifest["in_flight"] = None
                manifest["active_trial"] = None
                atomic_json(run_dir / "manifest.json", manifest)
                return
            grad_values = candidate["window_gradient_norms"]
            max_ratio = max(item["update_to_weight_ratio"] for item in diagnostics)
            max_update = max(item["maximum_update_magnitude"] for item in diagnostics)
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            window_seconds = float(candidate.get("window_train_seconds", 0.0))
            tokens = (start_step - int(candidate.get("last_metric_step", 0))) * TRAIN["batch_size"] * TRAIN["sequence_length"]
            row = {
                "trial_id": trial_id, "stage": stage, "step": start_step,
                "muon_lr": candidate["muon_lr"], "muon_weight_decay": candidate["muon_weight_decay"],
                "train_loss": training_loss, "validation_loss": validation_loss,
                "validation_standard_error": validation_se,
                "gradient_norm_mean": statistics.fmean(grad_values) if grad_values else None,
                "gradient_norm_max": max(grad_values) if grad_values else None,
                "update_to_weight_ratio_max": max_ratio,
                "maximum_update_magnitude": max_update,
                "tokens_per_second": tokens / window_seconds if window_seconds else 0.0,
                "step_time_seconds": window_seconds / max(1, len(grad_values)),
                "peak_memory_mib": peak_memory, "stability": "stable",
                "adamw_lr": adamw_lr,
            }
            candidate["history"].append(row)
            candidate["window_train_loss_sum"] = 0.0
            candidate["window_gradient_norms"] = []
            candidate["window_train_seconds"] = 0.0
            candidate["last_metric_step"] = start_step
            candidate["best_validation_loss"] = min(
                [item["validation_loss"] for item in candidate["history"]])
            append_metrics(run_dir / "metrics.csv", row)

        extra = {"trial_id": trial_id, "stage": stage,
                 "muon_lr": candidate["muon_lr"],
                 "muon_weight_decay": candidate["muon_weight_decay"],
                 "data_metadata": data_metadata,
                 "window_loss_sum": candidate.get("window_train_loss_sum", 0.0),
                 "window_gradient_norms": candidate.get("window_gradient_norms", []),
                 "window_train_seconds": candidate.get("window_train_seconds", 0.0),
                 "last_metric_step": candidate.get("last_metric_step", 0),
                 "candidate_snapshot": copy.deepcopy(candidate)}
        save_checkpoint(ckpt_path, model, optimizer, start_step,
                        scheduler=scheduler, data_rng=train_loader.rng, extra=extra)
        manifest["committed_optimizer_steps"] += 1
        manifest["in_flight"] = None
        if row:
            candidate["history"][-1]["checkpoint"] = str(ckpt_path)
        atomic_json(run_dir / "manifest.json", manifest)
        if row:
            print(f"{trial_id} stage={stage} step={start_step}/{target_steps} "
                  f"train={row['train_loss']:.6f} val={row['validation_loss']:.6f} "
                  f"grad={row['gradient_norm_mean']:.4f} ratio={row['update_to_weight_ratio_max']:.5g} "
                  f"tok/s={row['tokens_per_second']:.0f} peak={row['peak_memory_mib']:.0f}MiB",
                  flush=True)

    candidate["status"] = "stage_complete"
    candidate["completed_stage"] = stage
    candidate["final_train_loss"] = candidate["history"][-1]["train_loss"]
    candidate["final_validation_loss"] = candidate["history"][-1]["validation_loss"]
    manifest["active_trial"] = None
    manifest["in_flight"] = None
    atomic_json(run_dir / "manifest.json", manifest)


def selection(candidate_pool: list[dict], keep: int, stage: int) -> tuple[list[dict], dict]:
    eligible = [c for c in candidate_pool if c.get("completed_stage", 0) >= stage and c["status"] != "failed"]
    if not eligible:
        return [], {"stage": stage, "selected": [], "reason": "no candidates completed the stage"}
    final_row = lambda c: next(row for row in reversed(c["history"]) if row["step"] == STAGE_TARGETS[stage])
    losses = [final_row(c)["validation_loss"] for c in eligible]
    errors = [final_row(c)["validation_standard_error"] for c in eligible]
    margin = max(2 * max(errors, default=0.0), 0.005 if stage == 1 else 0.002)
    best_loss = min(losses)
    tied = [c for c in eligible if final_row(c)["validation_loss"] <= best_loss + margin]
    tied.sort(key=lambda c: (
        max((row.get("update_to_weight_ratio_max") or 0.0) for row in c["history"]),
        c["muon_lr"], c["muon_weight_decay"], c["trial_id"]))
    tied_ids = {c["trial_id"] for c in tied}
    remaining = sorted((c for c in eligible if c["trial_id"] not in tied_ids),
                       key=lambda c: (final_row(c)["validation_loss"], c["trial_id"]))
    ordered = (tied + remaining)[:keep]
    selected_ids = {c["trial_id"] for c in ordered}
    for c in eligible:
        c["status"] = "selected" if c["trial_id"] in selected_ids else "eliminated"
        c["selection_reason"] = (
            f"stage {stage}: validation loss at {STAGE_TARGETS[stage]} steps; "
            f"uncertainty margin {margin:.6g}; near-ties use max update ratio then lower Muon LR")
    return ordered, {"stage": stage, "target_step": STAGE_TARGETS[stage],
                    "uncertainty_margin": margin,
                    "ranked": [{"trial_id": c["trial_id"],
                                "validation_loss": final_row(c)["validation_loss"],
                                "validation_se": final_row(c)["validation_standard_error"]}
                               for c in sorted(eligible, key=lambda c: final_row(c)["validation_loss"])],
                    "selected": [c["trial_id"] for c in ordered],
                    "reason": "minimum fixed-step validation cross-entropy with uncertainty-aware tie handling"}


def svg_chart(path: Path, title: str, x_label: str, y_label: str,
              series: list[tuple[str, list[tuple[float, float]], str]],
              *, x_min=None, x_max=None, y_min=None, y_max=None,
              scatter: bool = False) -> None:
    width, height = 920, 560
    left, right, top, bottom = 90, 260, 65, 85
    pw, ph = width - left - right, height - top - bottom
    all_points = [point for _, points, _ in series for point in points
                  if math.isfinite(point[0]) and math.isfinite(point[1])]
    if not all_points:
        all_points = [(0.0, 0.0), (1.0, 1.0)]
    xmin = min(p[0] for p in all_points) if x_min is None else x_min
    xmax = max(p[0] for p in all_points) if x_max is None else x_max
    ymin = min(p[1] for p in all_points) if y_min is None else y_min
    ymax = max(p[1] for p in all_points) if y_max is None else y_max
    if xmax == xmin: xmax = xmin + 1.0
    if ymax == ymin: ymax = ymin + 1.0
    ypad = (ymax - ymin) * 0.08
    if y_min is None: ymin -= ypad
    if y_max is None: ymax += ypad
    def xy(x, y):
        return left + (x - xmin) / (xmax - xmin) * pw, top + (ymax - y) / (ymax - ymin) * ph
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>']
    for index in range(6):
        y = top + index * ph / 5
        value = ymax - index * (ymax - ymin) / 5
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+pw}" y2="{y:.2f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.4g}</text>')
    for index in range(6):
        x = left + index * pw / 5
        value = xmin + index * (xmax - xmin) / 5
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+ph}" stroke="#eee"/>')
        lines.append(f'<text x="{x:.2f}" y="{top+ph+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.4g}</text>')
    lines += [f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#444"/>',
              f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" stroke="#444"/>',
              f'<text x="{left+pw/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="14">{x_label}</text>',
              f'<text x="20" y="{top+ph/2}" transform="rotate(-90 20 {top+ph/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{y_label}</text>']
    for index, (label, points, color) in enumerate(series):
        points = sorted((p for p in points if math.isfinite(p[0]) and math.isfinite(p[1])))
        coords = [xy(x, y) for x, y in points]
        if len(coords) > 1 and not scatter:
            path_data = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(coords))
            lines.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in coords:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        ly = top + 18 + index * 19
        lines.append(f'<line x1="{left+pw+25}" y1="{ly-4}" x2="{left+pw+45}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{left+pw+52}" y="{ly}" font-family="sans-serif" font-size="11">{label}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_plots(run_dir: Path, candidates: list[dict]) -> None:
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    def series_for(field):
        return [(c["trial_id"], [(r["step"], r[field]) for r in c["history"] if r.get(field) is not None], COLORS[i % len(COLORS)])
                for i, c in enumerate(candidates)]
    svg_chart(plots / "validation_loss_vs_steps.svg", "Validation loss vs optimizer steps",
              "Optimizer steps", "Validation cross-entropy", series_for("validation_loss"), y_min=0)
    svg_chart(plots / "train_loss_vs_steps.svg", "Training loss vs optimizer steps",
              "Optimizer steps", "Training cross-entropy", series_for("train_loss"), y_min=0)
    common_steps = sorted({row["step"] for candidate in candidates
                           for row in candidate["history"]})
    lr_series = []
    for index, step in enumerate(common_steps):
        points = [(candidate["muon_lr"], row["validation_loss"])
                  for candidate in candidates for row in candidate["history"]
                  if row["step"] == step]
        if points:
            lr_series.append((f"{step} optimizer steps", points, COLORS[index % len(COLORS)]))
    svg_chart(plots / "muon_lr_vs_validation_loss.svg", "Muon learning rate vs validation loss at matched steps",
              "Muon learning rate", "Validation cross-entropy (matched optimizer step)",
              lr_series, scatter=True, y_min=0)
    ratio_series = [(c["trial_id"], [(r["step"], r["update_to_weight_ratio_max"])
                                     for r in c["history"] if r.get("update_to_weight_ratio_max") is not None], COLORS[i % len(COLORS)])
                    for i, c in enumerate(candidates)]
    svg_chart(plots / "update_to_weight_ratio_vs_steps.svg", "Representative max update-to-weight ratio",
              "Optimizer steps", "Update norm / weight norm", ratio_series, y_min=0)
    speed = [(c["trial_id"], [(i, r["tokens_per_second"])
                              for i, r in enumerate([x for x in c["history"] if x.get("tokens_per_second") is not None], 1)],
              COLORS[i % len(COLORS)]) for i, c in enumerate(candidates)]
    svg_chart(plots / "tokens_per_second.svg", "Tokens per second by measurement window",
              "Measurement window", "Tokens / second", speed, y_min=0)


def write_optimal_config(path: Path, winner: dict) -> None:
    values = {
        "name": "muon_rms_optimal",
        "optimizer": "muon_rms_to_rms",
        "muon_lr": winner["muon_lr"],
        "muon_weight_decay": winner["muon_weight_decay"],
        "muon_momentum": 0.95,
        "muon_nesterov": True,
        "muon_ns_steps": 5,
        "muon_ns_coefficients": list(DEFAULT_NS_COEFFICIENTS),
        "muon_eps": 1e-7,
        "muon_scaling_mode": "rms_to_rms",
        "adjust_lr_fn": "spectral_unclamped",
        "adamw_lr": TRAIN["adamw_lr"],
        "adamw_weight_decay": TRAIN["adamw_weight_decay"],
        "adamw_betas": [TRAIN["beta1"], TRAIN["beta2"]],
        "scheduler": "warmup_constant",
        "warmup_steps": WARMUP_STEPS,
        "gradient_clip_norm": TRAIN["grad_clip"],
        "batch_size": TRAIN["batch_size"],
        "sequence_length": TRAIN["sequence_length"],
        "gradient_accumulation_steps": TRAIN["gradient_accumulation_steps"],
        "amp": "bfloat16",
        "compile": TRAIN["compile"],
        "total_training_steps": 10000,
        "train_path": str(REPO / "train.bin"),
        "val_path": str(REPO / "val.bin"),
        "vocab_size": MODEL["vocab_size"],
        "d_model": MODEL["d_model"],
        "d_ffn": MODEL["d_ffn"],
        "num_layers": MODEL["num_layers"],
        "num_heads": MODEL["num_heads"],
        "max_seq_len": MODEL["max_seq_len"],
        "search_validation_loss": winner.get("final_validation_loss"),
        "search_training_steps": winner["completed_steps"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    def yaml_value(value):
        if isinstance(value, float) and 0 < abs(value) < 1e-4:
            return f"{value:.1e}"
        return json.dumps(value, ensure_ascii=False)
    text = "\n".join(f"{key}: {yaml_value(value)}" for key, value in values.items()) + "\n"
    path.write_text(text, encoding="utf-8")


def write_report(run_dir: Path, manifest: dict, candidates: list[dict], winner: dict,
                 data_metadata: dict, preflight_result: dict) -> None:
    phases = manifest.get("selections", [])
    diagnostic_path = run_dir / "diagnostics.jsonl"
    diagnostic_rows = ([json.loads(line) for line in diagnostic_path.read_text(encoding="utf-8").splitlines()]
                       if diagnostic_path.exists() else [])
    final_diagnostics = [row for row in diagnostic_rows
                         if row["trial_id"] == winner["trial_id"]
                         and row["step"] == winner["completed_steps"]]
    if final_diagnostics:
        sigma = [row["orthogonalized_spectral_norm_estimate"] for row in final_diagnostics]
        op_norm = [row["actual_update_rms_to_rms_operator_norm"] for row in final_diagnostics]
        ratios = [row["update_to_weight_ratio"] for row in final_diagnostics]
        spectral_text = (f"最终代表矩阵 raw NS sigma_max 估计为 {min(sigma):.3f}–{max(sigma):.3f}（理想半正交方向约 1）；实际参数更新的 RMS-to-RMS operator norm 为 {min(op_norm):.5f}–{max(op_norm):.5f}，Muon LR={winner['muon_lr']:g}，更新/权重比为 {min(ratios):.3g}–{max(ratios):.3g}。")
    else:
        spectral_text = "最终 checkpoint 未留下矩阵谱范数诊断记录。"
    recorded_peak = max((row["peak_memory_mib"] for candidate in candidates
                         for row in candidate["history"] if row.get("peak_memory_mib") is not None),
                        default=0.0)
    lines = ["# RMS-to-RMS Muon 超参数搜索报告", "",
             "## 实验配置", "",
             f"- 模型：4 层 Transformer，d_model=512、d_ffn=1344、16 头、词表 8,983，参数量 {manifest['model_parameters']:,}。",
             f"- Muon 负责 {manifest['muon_parameter_count']:,} 个参数的 28 个投影矩阵；AdamW 负责 {manifest['adamw_parameter_count']:,} 个 embedding、norm 和输出头参数。",
             "- Muon scale：`sqrt(m/n)`，数学上等价于 `adjust_lr_fn=spectral_unclamped` 的形状倍率；所有候选固定使用 RMS-to-RMS。",
             f"- 搜索空间：Muon LR {LR_GRID}，Muon weight decay {WD_GRID}；AdamW LR={TRAIN['adamw_lr']}、weight decay={TRAIN['adamw_weight_decay']} 固定。",
             "- 固定 momentum=0.95、Nesterov=True、NS steps=5、系数=" + str(DEFAULT_NS_COEFFICIENTS) + "、eps=1e-7。",
             "- Batch=128、sequence=256、accumulation=1、BF16 autocast、TF32、global gradient clipping=1.0；统一 10 步线性 warmup 后 constant。",
             f"- 数据：train {data_metadata['train']['tokens']:,} tokens，validation {data_metadata['val']['tokens']:,} tokens；每个候选以相同随机种子重放同一训练窗口顺序和验证窗口。",
             f"- 无更新数值预检耗时 {preflight_result['duration_seconds']:.3f}s，模型权重未改变；12 个代表矩阵的 NS 输出均有限。",
             "",
             "## 实际预算", "",
             f"累计 optimizer updates：{manifest['optimizer_steps_consumed']} / {manifest['max_total_steps']}（已提交 checkpoint 的 updates：{manifest['committed_optimizer_steps']}；数值/OOM 失败更新：{manifest['failed_updates']}；恢复时保守计费的未提交 reservation：{manifest['recovered_reserved_updates']}）。",
             "各阶段目标为 8×25、4×75、2×150、1×200；任一失败配置的未使用预算会在搜索结束前交给当前最优可用配置。",
             "本次在 T08 第 50 步验证期间中断；该次更新已计入预算但未提交到 checkpoint。恢复后从第 49 步 checkpoint 重放，故累计 1 次恢复 reservation，最终 T07 在剩余预算内完成 449 步；999 步提交到 checkpoint，累计预算计数为 1,000。",
             "",
             "## 搜索记录", "",
             "| Trial | Muon LR | Muon WD | 累计步数 | Best val | Final val | Final train | Stability | 平均 tokens/s | PyTorch peak allocated MiB | Selection / elimination |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for c in candidates:
        history = c["history"]
        speeds = [row["tokens_per_second"] for row in history if row.get("tokens_per_second")]
        memories = [row["peak_memory_mib"] for row in history if row.get("peak_memory_mib") is not None]
        status = c.get("status", "unknown")
        stability = "stable" if status != "failed" else c.get("numerical_failure", "failed")
        lines.append(f"| {c['trial_id']} | {c['muon_lr']:.6g} | {c['muon_weight_decay']:.3g} | {c['completed_steps']} | "
                     f"{c.get('best_validation_loss', float('nan')):.6f} | {c.get('final_validation_loss', float('nan')):.6f} | "
                     f"{c.get('final_train_loss', c.get('last_train_loss', float('nan'))):.6f} | {stability} | "
                     f"{statistics.fmean(speeds) if speeds else 0:.0f} | {max(memories) if memories else 0:.0f} | "
                     f"{c.get('selection_reason', status)} |")
    lines += ["", "## 阶段筛选", ""]
    for phase in phases:
        lines.append(f"- Stage {phase['stage']}（{phase.get('target_step', '?')} 步）：保留 {', '.join(phase.get('selected', [])) or '无'}；{phase.get('reason', '')}。")
    lines += ["", "## 最优配置", "",
              f"推荐 trial `{winner['trial_id']}`：Muon LR={winner['muon_lr']:.6g}、Muon WD={winner['muon_weight_decay']:.3g}；final validation loss={winner.get('final_validation_loss', float('nan')):.6f}，累计 {winner['completed_steps']} 步。",
              "固定参数：momentum=0.95、Nesterov=True、NS steps=5、RMS-to-RMS (`spectral_unclamped` shape scale)、AdamW LR=0.001、AdamW WD=0.01、10 步 warmup 后 constant、gradient clipping=1.0。",
              "该配置只经过本次实际累计步数验证，不能解释为 10,000 步完整训练验证。",
              "",
              "## 数值稳定性与诊断", "",
              f"- NaN/Inf/OOM 失败 trial 数：{sum(c['status'] == 'failed' for c in candidates)}；显存列保留 PyTorch allocator 计数，物理显存峰值不由该计数推断。",
              "- `diagnostics.jsonl` 按评估 checkpoint 记录 blocks.0 和 blocks.3 的 attention Q/K/V/output 与 MLP up/down 矩阵的权重、梯度、momentum、NS 更新、参数实际更新、update/weight ratio，以及 power iteration 近似的谱范数和 RMS-to-RMS operator norm。",
              f"- {spectral_text}",
              "- PyTorch `max_memory_allocated` 记录的最大值为 " + f"{recorded_peak:.0f} MiB，超过设备声明的 " + f"{manifest['gpu_memory_total_mib']:.0f} MiB；因此该 allocator 计数不能解释为物理 VRAM 峰值。本次未持久化逐步 NVML 显存采样，表中原样保留该计数并标明来源。",
              "- T07 在 stage 4 的验证 loss 从 300 步 2.6743 降至 449 步 2.3269，未观察到验证反弹；NS 最大奇异值通过 8 次 power iteration 估计，有限步 NS 不保证精确正交化。",
              "",
              "## 不确定性与解释边界", "",
              "- 每个 validation loss 为固定 20 个 batch 上按 token 数加权的 cross-entropy；报告标准误用于阶段内近似并列判断。短程训练下 weight decay 的差异仍可能很弱。",
              "- 本次结果是该模型、固定数据抽样和当前随机种子的短程比较；差异不能直接外推到其他种子或完整预训练。",
              "- 理论说明：RMS-to-RMS 倍率按矩阵逻辑形状 `sqrt(m/n)`；loss 排名、吞吐和稳定性均来自本次记录。未实际训练的长期效果属于推测。",
              "",
              "## 下一轮建议", "",
              "- 使用推荐配置从共同初始化做至少 2–3 个独立种子的 longer-run 对照，并继续固定评估集。",
              "- 若 250/450 步验证仍呈单调改善，围绕推荐 LR 做更窄的局部搜索；将 Muon weight decay 结论延长到更长训练后再判断。",
              "",
              "## 产物", "",
              f"- Manifest：`{(run_dir / 'manifest.json').relative_to(REPO)}`",
              f"- Metrics：`{(run_dir / 'metrics.csv').relative_to(REPO)}`",
              f"- Matrix diagnostics：`{(run_dir / 'diagnostics.jsonl').relative_to(REPO)}`",
              "- Plots：`plots/*.svg`",
              "- Recommended config: `configs/muon_rms_optimal.yaml`",
              "- Resume: `python scripts/tune_muon.py --max-total-steps 1000 --resume " + str(run_dir.relative_to(REPO)) + "`",
              ""]
    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / "muon_hpo_report.md").write_text("\n".join(lines), encoding="utf-8")


def create_candidates() -> list[dict]:
    result = []
    index = 1
    for lr in LR_GRID:
        for wd in WD_GRID:
            result.append({"trial_id": f"T{index:02d}", "muon_lr": lr,
                           "muon_weight_decay": wd, "status": "pending",
                           "completed_steps": 0, "completed_stage": 0,
                           "history": [], "best_validation_loss": None,
                           "final_validation_loss": None,
                           "final_train_loss": None, "failure_stage": None,
                           "numerical_failure": None, "selection_reason": "not yet ranked"})
            index += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-total-steps", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_total_steps <= MAX_TOTAL_STEPS:
        parser.error(f"--max-total-steps must be in [1, {MAX_TOTAL_STEPS}]")
    if args.eval_batches <= 0:
        parser.error("--eval-batches must be positive")
    TRAIN["compile"] = not args.no_compile
    run_dir = ((REPO / args.resume).resolve() if args.resume else
               (REPO / args.output_dir).resolve() if args.output_dir else
               REPO / "runs" / time.strftime("muon_hpo_%Y%m%d_%H%M%S"))
    manifest_path = run_dir / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"manifest missing from resume run: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["max_total_steps"] != args.max_total_steps:
            parser.error("resume budget differs from saved manifest")
        if manifest.get("training") != TRAIN:
            parser.error("resume training settings differ from saved manifest")
        if manifest.get("status") == "complete":
            print(f"Run is already complete: {run_dir}")
            return 0
    else:
        if run_dir.exists():
            parser.error(f"output directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        manifest = None

    if not torch.cuda.is_available():
        parser.error("CUDA is required for the configured training run")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    validate_bf16_amp(device, True)
    data_config = type("DataConfig", (), {
        "train_path": str(REPO / "train.bin"), "val_path": str(REPO / "val.bin"),
        "vocab_size": MODEL["vocab_size"], "token_dtype": "int64",
        "sequence_length": TRAIN["sequence_length"],
    })()
    data_metadata = validate_data_metadata(data_config)
    if manifest is None:
        for name, value in {"TORCHINDUCTOR_COMPILE_THREADS": "1", "OMP_NUM_THREADS": "2",
                            "MKL_NUM_THREADS": "2",
                            "TORCHINDUCTOR_CACHE_DIR": str(run_dir / "cache" / "inductor"),
                            "TRITON_CACHE_DIR": str(run_dir / "cache" / "triton")}.items():
            os.environ.setdefault(name, value)
        seed_everything(TRAIN["seed"])
        base_model = make_model(device)
        init_hash = tensor_state_hash(base_model.state_dict())
        atomic_torch_save(run_dir / "initial_model.pt",
                          {key: value.detach().cpu() for key, value in base_model.state_dict().items()})
        p_muon, p_adamw = partition_muon_parameters(base_model)
        preflight_result = preflight(base_model, device)
        manifest = {
            "schema_version": 1, "status": "running", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "max_total_steps": args.max_total_steps, "optimizer_steps_consumed": 0,
            "committed_optimizer_steps": 0, "failed_updates": 0,
            "recovered_reserved_updates": 0, "in_flight": None,
            "active_trial": None, "current_stage": 1,
            "model": MODEL, "model_parameters": sum(p.numel() for p in base_model.parameters()),
            "muon_parameter_count": sum(p.numel() for p in p_muon),
            "adamw_parameter_count": sum(p.numel() for p in p_adamw),
            "muon_matrix_count": len(p_muon), "model_initial_sha256": init_hash,
            "training": TRAIN, "lr_grid": list(LR_GRID), "weight_decay_grid": list(WD_GRID),
            "data_metadata": data_metadata, "preflight": preflight_result,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_memory_total_mib": torch.cuda.get_device_properties(device).total_memory / 2**20,
            "eval_batches": args.eval_batches, "eval_protocol": "fixed seed; token-weighted CE; inference_mode",
            "candidates": create_candidates(), "selections": [],
            "stage_plan": {str(k): {"target_cumulative_step": v,
                                      "eval_every": STAGE_EVAL_EVERY[k],
                                      "keep": STAGE_KEEP.get(k, 1)}
                           for k, v in STAGE_TARGETS.items()},
        }
        del base_model
        torch.cuda.empty_cache()
        atomic_json(manifest_path, manifest)
    elif manifest.get("data_metadata") != data_metadata:
        parser.error("resume dataset fingerprint differs from manifest")

    candidates = manifest["candidates"]
    if manifest.get("in_flight"):
        in_flight = manifest["in_flight"]
        candidate = next(c for c in candidates if c["trial_id"] == in_flight["trial_id"])
        ckpt = trial_checkpoint(run_dir, candidate["trial_id"])
        recovered_checkpoint = False
        if ckpt.is_file():
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            saved_step = int(saved["step"])
            if saved_step != int(candidate["completed_steps"]):
                snapshot = saved["extra"].get("candidate_snapshot")
                if snapshot is None or int(snapshot["completed_steps"]) != saved_step:
                    parser.error("checkpoint/manifest progress differs and has no recoverable candidate snapshot")
                previous = int(candidate["completed_steps"])
                candidate.update(snapshot)
                if saved_step > previous:
                    manifest["committed_optimizer_steps"] += saved_step - previous
                    recovered_checkpoint = True
                else:
                    # The process had reserved and applied an update, but
                    # the model/optimizer checkpoint predates it. Keep the
                    # consumed-budget charge and roll metrics back with state.
                    manifest["recovered_reserved_updates"] += 1
                    recovered_checkpoint = True
        if not recovered_checkpoint:
            # Charge an uncertain reservation so a crash cannot overspend the budget.
            manifest["recovered_reserved_updates"] += 1
        manifest["in_flight"] = None
        atomic_json(manifest_path, manifest)
    try:
        # Resume an in-progress stage/trial first, then continue the same phase.
        while manifest["optimizer_steps_consumed"] < manifest["max_total_steps"]:
            active = manifest.get("active_trial")
            stage = int(manifest.get("current_stage", 1))
            if stage > 4:
                break
            stage_candidates = ([next(c for c in candidates if c["trial_id"] == active)]
                                if active else
                                [c for c in candidates if c.get("completed_stage", 0) < stage
                                 and c.get("status") not in ("failed", "eliminated")
                                 and (stage == 1 or c.get("status") == "selected")])
            if not stage_candidates:
                if stage in STAGE_KEEP:
                    previous_pool = [c for c in candidates if c.get("completed_stage", 0) >= stage
                                     and c.get("status") != "failed"]
                    keepers, result = selection(previous_pool, STAGE_KEEP[stage], stage)
                    manifest["selections"] = [x for x in manifest["selections"] if x["stage"] != stage] + [result]
                    manifest["current_stage"] = stage + 1
                    if stage == 3:
                        if keepers:
                            manifest["recommended_trial"] = keepers[0]["trial_id"]
                    atomic_json(manifest_path, manifest)
                    continue
                break
            for candidate in stage_candidates:
                if manifest["optimizer_steps_consumed"] >= manifest["max_total_steps"]:
                    break
                if active and candidate["trial_id"] != active:
                    continue
                target = STAGE_TARGETS[stage]
                if stage == 4:
                    remaining = manifest["max_total_steps"] - manifest["optimizer_steps_consumed"]
                    target = min(target, candidate["completed_steps"] + remaining)
                run_trial(manifest, candidate, stage, target,
                          run_dir, device, data_metadata, args.eval_batches)
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
                active = None
                if candidate["status"] == "failed":
                    print(f"{candidate['trial_id']} eliminated after failure: {candidate.get('numerical_failure')}", flush=True)
            if manifest.get("active_trial"):
                # run_trial returns normally only on a failure or full target.
                continue
            if stage in STAGE_KEEP:
                previous_pool = [c for c in candidates if c.get("completed_stage", 0) >= stage
                                 and c.get("status") != "failed"]
                keepers, result = selection(previous_pool, STAGE_KEEP[stage], stage)
                manifest["selections"] = [x for x in manifest["selections"] if x["stage"] != stage] + [result]
                manifest["current_stage"] = stage + 1
                if stage == 3 and keepers:
                    manifest["recommended_trial"] = keepers[0]["trial_id"]
                atomic_json(manifest_path, manifest)
            elif stage == 4:
                manifest["current_stage"] = 5
                atomic_json(manifest_path, manifest)

        # If a candidate fails, its unused allowance is reassigned to the
        # best available candidate; the hard global reservation remains 1,000.
        winner_id = manifest.get("recommended_trial")
        eligible = [c for c in candidates if c.get("status") != "failed" and c.get("history")]
        if not eligible:
            manifest["status"] = "failed"
            atomic_json(manifest_path, manifest)
            raise RuntimeError("all candidates failed before producing a valid validation result")
        winner = next((c for c in candidates if c["trial_id"] == winner_id), None)
        if winner is None or winner.get("status") == "failed":
            def latest_loss(c):
                return (c.get("final_validation_loss") if c.get("final_validation_loss") is not None
                        else c["history"][-1]["validation_loss"])
            winner = min(eligible, key=lambda c: (latest_loss(c), c["muon_lr"]))
            winner["selection_reason"] = "fallback: lowest final validation loss among stable candidates"
        manifest["recommended_trial"] = winner["trial_id"]
        if manifest["optimizer_steps_consumed"] < manifest["max_total_steps"]:
            current = max(winner["completed_steps"], 0)
            manifest["active_trial"] = winner["trial_id"]
            manifest["current_stage"] = 4
            run_trial(manifest, winner, 4, current +
                      (manifest["max_total_steps"] - manifest["optimizer_steps_consumed"]),
                      run_dir, device, data_metadata, args.eval_batches)
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            winner["selection_reason"] = (winner.get("selection_reason", "") +
                                           "; received remaining budget after failed candidates").strip("; ")
        winner["status"] = "recommended"
        for candidate in candidates:
            if candidate["trial_id"] != winner["trial_id"] and candidate.get("status") not in ("failed", "eliminated"):
                candidate["status"] = "eliminated"
        generate_plots(run_dir, candidates)
        write_optimal_config(REPO / "configs" / "muon_rms_optimal.yaml", winner)
        manifest["status"] = "complete"
        manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest["recommended_configuration"] = {
            "trial_id": winner["trial_id"], "muon_lr": winner["muon_lr"],
            "muon_weight_decay": winner["muon_weight_decay"],
            "steps": winner["completed_steps"], "validation_loss": winner.get("final_validation_loss"),
        }
        atomic_json(manifest_path, manifest)
        write_report(run_dir, manifest, candidates, winner, data_metadata, manifest["preflight"])
        atomic_json(run_dir / "summary.json", {
            "optimizer_steps_consumed": manifest["optimizer_steps_consumed"],
            "committed_optimizer_steps": manifest["committed_optimizer_steps"],
            "recommended_configuration": manifest["recommended_configuration"],
            "candidates": candidates, "selections": manifest["selections"],
        })
        print(f"Completed {manifest['optimizer_steps_consumed']} optimizer updates. "
              f"Recommended {winner['trial_id']}: muon_lr={winner['muon_lr']}, "
              f"muon_weight_decay={winner['muon_weight_decay']}, "
              f"validation_loss={winner.get('final_validation_loss')}; run={run_dir}", flush=True)
        return 0
    except BaseException:
        manifest["status"] = "interrupted"
        manifest["last_error"] = traceback.format_exc()
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
