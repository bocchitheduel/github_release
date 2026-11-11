"""Measure cached KV-owned backward kernels, optionally sweeping tile configs."""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton

import block_attention_triton as attention
from scripts.benchmark_attention_autotune import bench_cached, bench_fwd_bwd_cudagraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--rep-ms", type=int, default=10)
    parser.add_argument("--tile-cap", type=int, default=128)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--transposed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__} triton={triton.__version__}", flush=True)

    for length in args.lengths:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.manual_seed(length)
            shape = (1, length, 8, 64) if args.transposed else (1, 8, length, 64)
            inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            if args.transposed:
                inputs = [x.transpose(1, 2) for x in inputs]
            q, k, v = [x.requires_grad_() for x in inputs]
            grad = torch.randn_like(q)

        def step():
            out = attention.block_causal_attention_triton(q, k, v, args.tile_cap, args.tile_cap)
            return out, torch.autograd.grad(out, (q, k, v), grad)

        total_ms = bench_fwd_bwd_cudagraph(step, stream, args.rep_ms)
        autotuner = attention._attention_backward_kv_outer_kernel
        best = autotuner.best_config
        saved = []
        with torch.cuda.stream(stream):
            with torch.autograd.graph.saved_tensors_hooks(lambda x: saved.append(x) or x, lambda x: x):
                out = attention.block_causal_attention_triton(q, k, v, args.tile_cap, args.tile_cap)
            qk, kk, vk, output, lse = saved
            do = grad.reshape(8, length, 64)
            delta = torch.empty((8, length), device="cuda", dtype=torch.float32)
            dq_acc = torch.zeros(q.numel(), device="cuda", dtype=torch.float32)
            dq, dk, dv = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v)]
        stream.synchronize()
        batch, heads, *q_strides = attention._attention_batch_layout(qk)
        k_strides = attention._attention_batch_layout(kk)[2:]
        v_strides = attention._attention_batch_layout(vk)[2:]

        def run_delta():
            return attention._attention_delta_kernel[(triton.cdiv(length, 16), batch)](
                output, do, delta, dq_acc, length, 64, 64, batch, *do.stride(),
                block_m=16, block_d=64, block_dv=64, num_warps=4,
            )

        delta_ms = bench_cached(run_delta, args.rep_ms)
        cast_ms = bench_cached(lambda: attention._cast_fp32_gradient_kernel[(triton.cdiv(q.numel(), 1024),)](
            dq_acc, dq, q.numel(), block_size=1024, num_warps=4,
        ), args.rep_ms)

        def launch(config):
            return autotuner.fn[(triton.cdiv(length, config.kwargs["block_n"]), batch)](
                qk, kk, vk, output, lse, do, delta, dq_acc, dk, dv,
                length, length, batch, heads, 64, 64,
                *q_strides, *k_strides, *v_strides,
                1 / math.sqrt(64), "tf32", "tf32x3", "tf32", 2,
                *do.stride(), args.tile_cap, args.tile_cap,
                block_d=64, block_dv=64,
                **config.all_kwargs(),
            )

        configs = attention._KV_OUTER_CONFIGS if args.sweep else [best]
        for config in configs:
            if max(config.kwargs.values()) > args.tile_cap:
                continue
            try:
                compiled = launch(config)
                kernel_ms = bench_cached(lambda: launch(config), args.rep_ms)
                record = dict(
                    length=length, transposed=args.transposed,
                    block_m=config.kwargs["block_m"], block_n=config.kwargs["block_n"],
                    num_warps=config.num_warps, num_stages=config.num_stages,
                    best=config == best, fwd_bwd_us=total_ms * 1000,
                    main_us=kernel_ms * 1000, delta_us=delta_ms * 1000,
                    zero_us=0.0, cast_us=cast_ms * 1000,
                    registers=compiled.n_regs, spills=compiled.n_spills,
                    shared_bytes=compiled.metadata.shared,
                )
            except triton.OutOfResources as error:
                record = dict(length=length, config=str(config), error=str(error))
            records.append(record)
            print(json.dumps(record), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
