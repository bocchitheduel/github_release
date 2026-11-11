"""Compare native LM-head cross entropy with Liger fused linear cross entropy.

Uses the project's 128 x 256 training shape, real token batches, SDPA, BF16
autocast, and the fused AdamW path. The two variants run in fresh processes.
No checkpoints are written.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(OUT / "cache/inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(OUT / "cache/triton"))


def emit(**values):
    print(json.dumps(values), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("native", "liger_fused", "suite"))
    parser.add_argument("--attention-backend", choices=("block", "sdpa"), default="sdpa")
    parser.add_argument("--optimizer", choices=("manual", "fused"), default="fused")
    args = parser.parse_args()
    if args.variant == "suite":
        prefix = f"{args.attention_backend}_{args.optimizer}"
        for variant in ("native", "liger_fused"):
            with (OUT / f"{prefix}_{variant}.jsonl").open("w") as stream, \
                    (OUT / f"{prefix}_{variant}.stderr").open("w") as errors:
                emit(event="starting_process", variant=variant)
                result = subprocess.run(
                    [sys.executable, "-u", __file__, variant,
                     "--attention-backend", args.attention_backend,
                     "--optimizer", args.optimizer],
                    stdout=stream, stderr=errors, check=False)
                emit(event="finished_process", variant=variant,
                     returncode=result.returncode)
                if result.returncode:
                    raise SystemExit(result.returncode)
        return

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from adamW import AdamW
    from crossentropy import cross_entropy
    from dataloader import DataLoader
    from train_utils import clip_grad_norm_
    from transformer_LM import TransformerLM

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    emit(event="environment", gpu=torch.cuda.get_device_name(0),
         torch=torch.__version__, variant=args.variant,
         attention_backend=args.attention_backend, optimizer=args.optimizer)

    fused = args.variant == "liger_fused"
    model = TransformerLM(
        vocab_size=8983, d_model=512, num_layers=4, num_heads=16,
        d_ffn=1344, max_seq_len=512, device="cuda",
        attention_backend=args.attention_backend,
        fused_lm_head_ce=fused)
    use_fused_optimizer = args.optimizer == "fused"
    optimizer = (torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
                 if use_fused_optimizer else AdamW(model.parameters(), lr=1e-3))
    if not fused:
        if use_fused_optimizer:
            loss_fn = torch.compile(
                lambda logits, targets: F.cross_entropy(
                    logits.flatten(0, 1), targets.flatten()))
        else:
            loss_fn = torch.compile(cross_entropy)
    model.compile()
    loader = DataLoader(ROOT / "train.bin", 128, 256, device="cuda", seed=42)
    inputs, targets = next(loader)
    emit(event="starting_compile", parameters=sum(p.numel() for p in model.parameters()))

    def step(load=False):
        nonlocal inputs, targets
        if load:
            inputs, targets = next(loader)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if fused:
                loss = model(inputs, targets=targets)
            else:
                loss = loss_fn(model(inputs), targets)
        loss.backward()
        if use_fused_optimizer:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=True, error_if_nonfinite=True)
        else:
            clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    def loss_for(inputs, targets):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if fused:
                return model(inputs, targets=targets)
            return loss_fn(model(inputs), targets)

    def memory_probe():
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        loss = loss_for(inputs, targets)
        torch.cuda.synchronize()
        forward_peak = torch.cuda.max_memory_allocated() / 2**20
        after_forward = torch.cuda.memory_allocated() / 2**20
        loss.backward()
        torch.cuda.synchronize()
        backward_peak = torch.cuda.max_memory_allocated() / 2**20
        clip = (torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, foreach=True, error_if_nonfinite=True)
                if use_fused_optimizer else clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        torch.cuda.synchronize()
        emit(event="memory_stages", forward_peak_mib=forward_peak,
             after_forward_mib=after_forward,
             backward_peak_mib=backward_peak,
             full_step_peak_mib=torch.cuda.max_memory_allocated() / 2**20,
             loss=float(loss.detach()), grad_norm=float(clip))

    def timing(fn, warmup=10, repeats=10, rounds=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - start) / repeats)
        ms = statistics.median(samples)
        return dict(ms=ms, samples_ms=samples,
                    tokens_per_second=32768000 / ms,
                    peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)

    attention_context = (sdpa_kernel(SDPBackend.FLASH_ATTENTION)
                         if args.attention_backend == "sdpa" else nullcontext())
    with attention_context:
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        emit(event="first_step_complete", seconds=time.perf_counter() - start)
        memory_probe()
        emit(event="fixed_batch_train_step", **timing(step))

        import copy
        data_rng_state = copy.deepcopy(loader.rng.bit_generator.state)
        for _ in range(80):
            next(loader)
        torch.cuda.synchronize()
        loader.rng.bit_generator.state = data_rng_state
        emit(event="train_step_with_loader", data_pages_prewarmed=True,
             **timing(lambda: step(True)))


if __name__ == "__main__":
    main()
