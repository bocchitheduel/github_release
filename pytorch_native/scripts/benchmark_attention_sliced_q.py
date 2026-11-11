"""Tune an explicit FA2-style Sliced-Q Gluon forward kernel vs local Triton and FA2."""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import triton

import block_attention_triton as local_attention
from benchmark_attention_autotune import (
    bench_cached,
    bench_fwd_bwd_cudagraph,
    config_string,
    load_flash_attn,
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
        help="run the local warp-layout sweep without importing official FlashAttention 2",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--q-blocks", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--kv-blocks", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--precision", choices=["bf16", "tf32", "tf32x3"], default="bf16")
    parser.add_argument(
        "--fixed-kernel-tile", type=int, nargs=2, metavar=("BM", "BN"),
        default=[64, 32],
        help="compare warp partitions at this direct-kernel tile (default: 64 32)",
    )
    parser.add_argument("--rep-ms", type=int, default=10)
    args = parser.parse_args()

    flash_attn_func = None if args.skip_fa2 else load_flash_attn(args.fa2_site)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if local_attention._attention_forward_sliced_q_kernel is None:
        raise RuntimeError("Triton Gluon is required for this benchmark")

    torch.backends.cuda.matmul.allow_tf32 = False
    batch, heads, dim = 1, 8, 64
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"dtype=bf16 B={batch} H={heads} D={dim} causal=True "
        f"precision={args.precision}"
    )
    print(
        "L,local_strategy,q_block,kv_block,local_fwd_ms,local_fwd_bwd_ms,"
        "FA2_fwd_ms,FA2_fwd_bwd_ms,local_fwd_config,local_bwd_config,"
        "same_tile_compiler_fwd_ms,same_tile_warps_m1_fwd_ms,"
        "same_tile_warps_m2_fwd_ms,same_tile_sliced_q_warps_m4_fwd_ms",
        flush=True,
    )

    for length in args.lengths:
        torch.manual_seed(17 + length)
        q0 = torch.randn(batch, length, heads, dim, device="cuda", dtype=torch.bfloat16)
        k0, v0 = torch.randn_like(q0), torch.randn_like(q0)
        qh0, kh0, vh0 = (
            x.transpose(1, 2).contiguous() for x in (q0, k0, v0)
        )
        if flash_attn_func is not None:
            fa2_fwd = lambda: flash_attn_func(q0, k0, v0, dropout_p=0.0, causal=True)
            fa2_fwd_ms = bench_cached(fa2_fwd, args.rep_ms)
        else:
            fa2_fwd_ms = float("nan")

        train_stream = torch.cuda.Stream()
        with torch.cuda.stream(train_stream):
            q = torch.randn(batch, length, heads, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
            k, v = torch.randn_like(q, requires_grad=True), torch.randn_like(q, requires_grad=True)
            grad = torch.randn_like(q)
            qh = torch.randn(batch, heads, length, dim, device="cuda", dtype=torch.bfloat16).requires_grad_()
            kh, vh = torch.randn_like(qh, requires_grad=True), torch.randn_like(qh, requires_grad=True)
            grad_h = torch.randn_like(qh)
        train_stream.synchronize()

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

        fixed_m, fixed_n = args.fixed_kernel_tile
        q_flat, k_flat, v_flat = [
            x.reshape(batch * heads, length, dim).contiguous()
            for x in (qh0, kh0, vh0)
        ]
        out_flat = torch.empty_like(q_flat)
        lse_flat = torch.empty((batch * heads, length), device="cuda", dtype=torch.float32)
        direct_grid = (triton.cdiv(length, fixed_m), batch * heads)
        kernel_args = (
            q_flat, k_flat, v_flat, out_flat, lse_flat,
            length, length, batch * heads, dim, dim, 1.0 / math.sqrt(dim),
        )

        def fixed_compiler_kernel():
            local_attention._attention_forward_kernel.fn[direct_grid](
                q_flat, k_flat, v_flat, out_flat, lse_flat,
                length, length, batch * heads, 1, dim, dim,
                *local_attention._attention_batch_layout(q_flat)[2:],
                *local_attention._attention_batch_layout(k_flat)[2:],
                *local_attention._attention_batch_layout(v_flat)[2:],
                1.0 / math.sqrt(dim),
                "tf32", args.precision, 2, fixed_m, fixed_n,
                block_m=fixed_m, block_n=fixed_n,
                block_d=dim, block_dv=dim,
                num_warps=4, num_stages=2,
            )

        fixed_compiler_ms = bench_cached(fixed_compiler_kernel, args.rep_ms)
        fixed_warp_ms = {}
        for warps_m in (1, 2, 4):
            def fixed_sliced_q_kernel(warps_m=warps_m):
                local_attention._attention_forward_sliced_q_kernel.fn[direct_grid](
                    *kernel_args, args.precision, 2, fixed_m, fixed_n,
                    warps_m=warps_m,
                    block_m=fixed_m, block_n=fixed_n,
                    block_d=dim, block_dv=dim,
                    num_warps=4, num_stages=2,
                )

            fixed_warp_ms[warps_m] = bench_cached(fixed_sliced_q_kernel, args.rep_ms)

        for q_block in args.q_blocks:
            for kv_block in args.kv_blocks:
                for strategy in ("compiler", "sliced_q"):
                    def local_forward():
                        with torch.no_grad():
                            return local_attention.block_causal_attention_triton(
                                qh0, kh0, vh0,
                                q_block_size=q_block,
                                kv_block_size=kv_block,
                                fp32_dot_precision=args.precision,
                                warp_strategy=strategy,
                            )

                    local_fwd_ms = bench_cached(local_forward, args.rep_ms)
                    forward_kernel = (
                        local_attention._attention_forward_sliced_q_kernel
                        if strategy == "sliced_q"
                        else local_attention._attention_forward_kernel
                    )
                    local_fwd_config = config_string(forward_kernel.best_config)
                    if strategy == "sliced_q":
                        warps_m = forward_kernel.best_config.kwargs["warps_m"]
                        local_fwd_config += f",mma_warps=[{warps_m},{4 // warps_m}]"

                    def local_fwd_bwd():
                        output = local_attention.block_causal_attention_triton(
                            qh, kh, vh,
                            q_block_size=q_block,
                            kv_block_size=kv_block,
                            fp32_dot_precision=args.precision,
                            warp_strategy=strategy,
                        )
                        grads = torch.autograd.grad(output, (qh, kh, vh), grad_h)
                        return output, grads

                    local_fwd_bwd_ms = bench_fwd_bwd_cudagraph(
                        local_fwd_bwd, train_stream, args.rep_ms
                    )
                    backward_config = config_string(
                        local_attention._attention_backward_kv_outer_kernel.best_config
                    )
                    print(
                        f"{length},{strategy},{q_block},{kv_block},"
                        f"{local_fwd_ms:.4f},{local_fwd_bwd_ms:.4f},"
                        f"{fa2_fwd_ms:.4f},{fa2_fwd_bwd_ms:.4f},"
                        f"{local_fwd_config},{backward_config},"
                        f"{fixed_compiler_ms:.4f},{fixed_warp_ms[1]:.4f},"
                        f"{fixed_warp_ms[2]:.4f},{fixed_warp_ms[4]:.4f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
