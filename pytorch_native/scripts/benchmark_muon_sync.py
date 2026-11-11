#!/usr/bin/env python3
"""Measure the host-visible cost of extra CUDA synchronization in Muon."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import muon_kernels
from muon import Muon


@contextmanager
def synchronize_after_muon_kernels():
    """Add a CUDA barrier after each Muon Triton wrapper and torch.bmm."""
    names = ("check_finite", "momentum_and_partial_norm", "normalize_direction",
             "polynomial_gemm_epilogue", "x_update_gemm_epilogue",
             "fused_parameter_update")
    originals = {name: getattr(muon_kernels, name) for name in names}
    original_bmm = torch.bmm

    def add_barrier(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            result = function(*args, **kwargs)
            torch.cuda.synchronize()
            return result
        return wrapped

    for name, function in originals.items():
        setattr(muon_kernels, name, add_barrier(function))

    @wraps(original_bmm)
    def bmm_with_barrier(*args, **kwargs):
        result = original_bmm(*args, **kwargs)
        torch.cuda.synchronize()
        return result

    torch.bmm = bmm_with_barrier
    try:
        yield
    finally:
        torch.bmm = original_bmm
        for name, function in originals.items():
            setattr(muon_kernels, name, function)


def make_optimizer():
    torch.manual_seed(391)
    shapes = ((512, 512),) * 4 + ((512, 1344),) * 2 + ((1344, 512),) * 2
    parameters = [torch.nn.Parameter(torch.randn(shape, device="cuda"))
                  for shape in shapes]
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer = Muon(parameters, lr=0.001, momentum=0.9, nesterov=True,
                     weight_decay=0.01, ns_steps=5, ns_precision="tf32x3")
    return optimizer


def host_work(milliseconds: float) -> None:
    """Run a short fixed-duration CPU task while queued CUDA work can proceed."""
    if milliseconds <= 0:
        return
    deadline = time.perf_counter() + milliseconds / 1000
    accumulator = 0
    while time.perf_counter() < deadline:
        accumulator = (accumulator * 1664525 + 1013904223) & 0xFFFFFFFF
    if accumulator == -1:  # keep the loop's work observable to Python.
        raise RuntimeError("unreachable")


def run_case(mode: str, steps: int, cpu_work_ms: float) -> dict:
    optimizer = make_optimizer()

    def step():
        optimizer.step()
        if mode == "step":
            torch.cuda.synchronize()
        host_work(cpu_work_ms)

    if mode == "kernel":
        context = synchronize_after_muon_kernels()
    else:
        @contextmanager
        def no_kernel_barriers():
            yield
        context = no_kernel_barriers()

    with context:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        for _ in range(steps):
            step()
        end_event.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        gpu_ms = float(start_event.elapsed_time(end_event))
    return {"mode": mode, "cpu_work_ms": cpu_work_ms, "steps": steps,
            "wall_ms_per_step": wall_ms / steps,
            "gpu_timeline_ms_per_step": gpu_ms / steps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20,
                        help="measured Muon steps per scenario")
    parser.add_argument("--cpu-work-ms", type=float, default=0.5,
                        help="synthetic host work per step, tested at 0 and this value")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "muon_sync_benchmark_20261110.md")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.steps <= 0 or args.cpu_work_ms < 0:
        parser.error("steps must be positive and cpu-work-ms must be nonnegative")

    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for cpu_ms in sorted(set((0.0, args.cpu_work_ms))):
        for mode in ("natural", "step", "kernel"):
            results.append(run_case(mode, args.steps, cpu_ms))

    lines = ["# Muon CUDA synchronization benchmark", "",
             f"- Date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"- GPU: {torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}",
             f"- Workload: 8 FP32 matrices, 5 Newton–Schulz steps, {args.steps} measured steps per case; TF32 disabled.",
             "- `natural` keeps Muon's existing finite-gradient check; `step` adds `torch.cuda.synchronize()` after every `Muon.step()`; `kernel` adds a barrier after each Muon Triton wrapper and `torch.bmm`.",
             "- CPU work is a synthetic fixed-duration host task between steps. `0 ms` isolates synchronization; the second setting tests whether queued GPU work overlaps host work.", "",
             "| Extra sync mode | Host work ms/step | Wall ms/step | CUDA event timeline ms/step | Wall slowdown vs natural |",
             "|---|---:|---:|---:|---:|"]
    natural = {row["cpu_work_ms"]: row["wall_ms_per_step"]
               for row in results if row["mode"] == "natural"}
    labels = {"natural": "Existing sync only", "step": "After every step",
              "kernel": "After each kernel"}
    for row in results:
        ratio = row["wall_ms_per_step"] / natural[row["cpu_work_ms"]]
        lines.append(f"| {labels[row['mode']]} | {row['cpu_work_ms']:.3f} | "
                     f"{row['wall_ms_per_step']:.3f} | "
                     f"{row['gpu_timeline_ms_per_step']:.3f} | {ratio:.3f}× |")
    lines.extend([
        "", "## Existing synchronization point", "",
        "The current batched CUDA Muon path reduces finite-gradient flags to a Python bool in `_step_batched_cuda`. That conversion synchronizes the CPU with CUDA once per `Muon.step()`. `natural` therefore means no additional barriers; it is not fully asynchronous.",
        "", "## Interpretation", "",
        "The after-step case measures the cost of waiting for the update kernels before returning from `step()`. The after-kernel case is intentionally diagnostic and inserts many barriers inside Newton–Schulz. Results are synthetic optimizer timings, not end-to-end training throughput.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
