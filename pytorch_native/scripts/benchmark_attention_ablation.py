"""Ablate FP32 dot precision and dK/dV accumulation in Triton attention."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import block_attention_triton as local_attention
from benchmark_attention_autotune import (
    bench_cached,
    bench_fwd_bwd_cudagraph,
    config_string,
    load_flash_attn,
)


def max_abs_diff(actual, reference):
    return max(
        (a.float() - r.float()).abs().max().item()
        for a, r in zip(actual, reference)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fa2-site",
        default="/tmp/cs336_fa2_bench_site_20261109",
        help="site-packages directory containing the official flash_attn wheel",
    )
    parser.add_argument(
        "--skip-fa2", action="store_true",
        help="run the local ablations without importing official FlashAttention 2",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--rep-ms", type=int, default=10)
    parser.add_argument(
        "--fixed-tile",
        type=int,
        default=None,
        help="force identical local query/KV tile caps (16 isolates the ablation modes)",
    )
    args = parser.parse_args()

    flash_attn_func = None if args.skip_fa2 else load_flash_attn(args.fa2_site)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    batch, heads, dim = 1, 8, 64
    variants = [
        ("tf32x3", "exclusive"),
        ("tf32x3", "atomic"),
        ("tf32", "atomic"),
        ("bf16", "atomic"),
        ("tf32x3", "partial"),
        ("tf32", "partial"),
        ("bf16", "partial"),
    ]
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"dtype=bf16 B={batch} H={heads} D={dim} causal=True"
    )
    print(
        "L,dot_precision,dkv_accumulation,local_fwd_ms,local_fwd_bwd_ms,"
        "FA2_fwd_ms,FA2_fwd_bwd_ms,output_diff_vs_baseline,grad_diff_vs_baseline,"
        "local_fwd_config,local_bwd_config"
    )

    for length in args.lengths:
        torch.manual_seed(17 + length)
        qh0 = torch.randn(batch, heads, length, dim, device="cuda", dtype=torch.bfloat16)
        kh0, vh0 = torch.randn_like(qh0), torch.randn_like(qh0)
        q0, k0, v0 = (x.transpose(1, 2).contiguous() for x in (qh0, kh0, vh0))

        if flash_attn_func is not None:
            fa2_fwd = lambda: flash_attn_func(q0, k0, v0, dropout_p=0.0, causal=True)
            fa2_fwd_ms = bench_cached(fa2_fwd, args.rep_ms)
        else:
            fa2_fwd_ms = float("nan")

        train_stream = torch.cuda.Stream()
        with torch.cuda.stream(train_stream):
            qh = torch.randn(
                batch, heads, length, dim,
                device="cuda", dtype=torch.bfloat16,
            ).requires_grad_()
            kh = torch.randn_like(qh, requires_grad=True)
            vh = torch.randn_like(qh, requires_grad=True)
            grad_h = torch.randn_like(qh)
            q, k, v = (
                x.detach().transpose(1, 2).contiguous().requires_grad_()
                for x in (qh, kh, vh)
            )
            grad = grad_h.transpose(1, 2).contiguous()
        train_stream.synchronize()

        def local_forward(precision):
            def run():
                with torch.no_grad():
                    return local_attention.block_causal_attention_triton(
                        qh0, kh0, vh0,
                        q_block_size=args.fixed_tile or 64,
                        kv_block_size=args.fixed_tile or 64,
                        fp32_dot_precision=precision,
                    )
            return run

        def local_step(precision, accumulation):
            def run():
                output = local_attention.block_causal_attention_triton(
                    qh, kh, vh,
                    q_block_size=args.fixed_tile or 64,
                    kv_block_size=args.fixed_tile or 64,
                    fp32_dot_precision=precision,
                    dkv_accumulation=accumulation,
                )
                grads = torch.autograd.grad(output, (qh, kh, vh), grad_h)
                return output, grads
            return run

        fwd_times = {}
        fwd_configs = {}
        for precision in ("tf32x3", "tf32", "bf16"):
            fwd_times[precision] = bench_cached(local_forward(precision), args.rep_ms)
            fwd_configs[precision] = config_string(
                local_attention._attention_forward_kernel.best_config
            )

        if flash_attn_func is not None:
            def fa2_fwd_bwd():
                output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
                grads = torch.autograd.grad(output, (q, k, v), grad)
                return output, grads

            fa2_fwd_bwd_ms = bench_fwd_bwd_cudagraph(
                fa2_fwd_bwd, train_stream, args.rep_ms
            )
        else:
            fa2_fwd_bwd_ms = float("nan")

        times = {}
        bwd_configs = {}
        steps = {}
        for precision, accumulation in variants:
            step = local_step(precision, accumulation)
            steps[(precision, accumulation)] = step
            times[(precision, accumulation)] = bench_fwd_bwd_cudagraph(
                step, train_stream, args.rep_ms
            )
            if accumulation == "exclusive":
                bwd_configs[(precision, accumulation)] = config_string(
                    local_attention._attention_backward_kv_outer_kernel.best_config
                )
            else:
                bwd_configs[(precision, accumulation)] = config_string(
                    local_attention._attention_backward_fused_kernel.best_config
                )

        baseline_step = steps[("tf32x3", "atomic")]
        with torch.cuda.stream(train_stream):
            baseline_output, baseline_grads = baseline_step()
            baseline_output = baseline_output.detach().clone()
            baseline_grads = tuple(grad_tensor.detach().clone() for grad_tensor in baseline_grads)
        train_stream.synchronize()

        for precision, accumulation in variants:
            with torch.cuda.stream(train_stream):
                output, grads = steps[(precision, accumulation)]()
                output_diff = max_abs_diff((output,), (baseline_output,))
                grad_diff = max_abs_diff(grads, baseline_grads)
            train_stream.synchronize()
            print(
                f"{length},{precision},{accumulation},"
                f"{fwd_times[precision]:.4f},{times[(precision, accumulation)]:.4f},"
                f"{fa2_fwd_ms:.4f},{fa2_fwd_bwd_ms:.4f},"
                f"{output_diff:.7g},{grad_diff:.7g},"
                f"{fwd_configs[precision]},{bwd_configs[(precision, accumulation)]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
