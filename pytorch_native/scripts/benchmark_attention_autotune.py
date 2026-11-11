"""Compare autotuned local Triton attention with official FlashAttention 2.

The first call for each shape compiles and autotunes outside the timed region.
Forward+backward uses an explicitly managed CUDA Graph stream. The benchmark
inputs are allocated on that stream too, so PyTorch autograd does not have to
insert a dependency from CUDA's legacy stream into the graph capture.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
import triton.testing

import block_attention_triton as local_attention


_gpu_warmup_state = None


def warm_gpu(duration_s=0.15):
    """Bring laptop GPU clocks up before measuring microsecond kernels."""
    global _gpu_warmup_state
    if _gpu_warmup_state is None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            left = torch.ones((2048, 2048), device="cuda", dtype=torch.float16)
            right = torch.ones_like(left)
            result = torch.empty_like(left)
            torch.mm(left, right, out=result)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(32):
                torch.mm(left, right, out=result)
        _gpu_warmup_state = stream, graph, left, right, result
    stream, graph, *_ = _gpu_warmup_state
    deadline = time.perf_counter() + duration_s
    with torch.cuda.stream(stream):
        while time.perf_counter() < deadline:
            graph.replay()
            stream.synchronize()


def load_flash_attn(site_path):
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func
    except ImportError:
        if site_path:
            sys.path.insert(0, site_path)
        from flash_attn import flash_attn_func
        return flash_attn_func


def config_string(config):
    if config is None:
        return "unavailable"
    tile = config.kwargs
    return (
        f"BM={tile['block_m']},BN={tile['block_n']},"
        f"warps={config.num_warps},stages={config.num_stages}"
    )


def bench_cached(fn, rep_ms):
    # Trigger Triton compilation/autotuning before timing and synchronize so
    # graph capture only sees cache hits and steady-state GPU work.
    fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    repeat = 128
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repeat):
            result = fn()
    return time_graph(graph, stream, repeat)


def time_graph(graph, stream, repeat):
    # Warm the actual workload long enough for laptop power/clock control to
    # settle. A few small replays immediately after a heavy GEMM warmup can
    # measure a different power state for each implementation.
    deadline = time.perf_counter() + 0.15
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    with torch.cuda.stream(stream):
        while time.perf_counter() < deadline:
            for _ in range(3):
                graph.replay()
            stream.synchronize()
        for _ in range(10):
            start.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            samples.append(start.elapsed_time(end) / repeat)
    return torch.tensor(samples).median().item()


def bench_fwd_bwd_cudagraph(fn, stream, rep_ms):
    """Capture and time one steady-state fwd+bwd step on ``stream``.

    ``fn``'s inputs must also have been allocated on ``stream``. This matters
    for autograd: with default-stream leaves, its stream bookkeeping may try
    to make the legacy stream wait on the stream currently being captured.
    """
    with torch.cuda.stream(stream):
        # Run lazy initialization and Triton autotuning before capture.
        for _ in range(3):
            fn()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured_result = fn()
    # Keep the captured outputs alive for the lifetime of the graph.
    _ = captured_result

    with torch.cuda.stream(stream):
        graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        graph.replay()
        end.record(stream)
    end.synchronize()
    estimate_ms = start.elapsed_time(end)
    repeat = min(128, max(64, int(rep_ms / max(estimate_ms, 1e-3))))

    # Capture multiple complete steps in one graph, amortizing Python launch
    # overhead. Capturing graph.replay() inside another graph is unsupported
    # by some PyTorch versions, so capture fn() directly on the input stream.
    batch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(batch_graph, stream=stream):
        for _ in range(repeat):
            captured_result = fn()

    return time_graph(batch_graph, stream, repeat)


def profile_kernel_times(fn, stream, steps=5):
    """Return average CUDA kernel durations, excluding CPU operator events."""
    warm_gpu()
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as profile:
        with torch.cuda.stream(stream):
            for _ in range(steps):
                result = fn()
        stream.synchronize()
    return [dict(name=event.key, us=event.device_time_total / steps)
            for event in profile.key_averages() if event.device_type.name == "CUDA"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fa2-site",
        default="/tmp/cs336_fa2_bench_site_20261109",
        help="site-packages directory containing the official flash_attn wheel",
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--rep-ms", type=int, default=20)
    parser.add_argument("--tile-cap", type=int, default=128)
    parser.add_argument("--output", type=Path, help="write timings, configs, and errors as JSON")
    parser.add_argument("--baseline-file", type=Path, help="also time an earlier local implementation")
    parser.add_argument("--profile-kernels", action="store_true")
    args = parser.parse_args()

    flash_attn_func = load_flash_attn(args.fa2_site)
    baseline = None
    if args.baseline_file:
        spec = importlib.util.spec_from_file_location("attention_baseline", args.baseline_file)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"triton={triton.__version__} dtype=bf16 B=1 H=8 D=64 causal=True"
    )
    print(
        "L, local_fwd_ms, FA2_fwd_ms, local_fwd_bwd_ms, FA2_fwd_bwd_ms, "
        "local_fwd_config, local_bwd_config"
    )

    batch, heads, dim = 1, 8, 64
    records = []
    for length in args.lengths:
        torch.manual_seed(17 + length)
        q0 = torch.randn(batch, length, heads, dim, device="cuda", dtype=torch.bfloat16)
        k0 = torch.randn_like(q0)
        v0 = torch.randn_like(q0)
        qh0, kh0, vh0 = (x.transpose(1, 2) for x in (q0, k0, v0))

        def local_fwd():
            with torch.no_grad():
                return local_attention.block_causal_attention_triton(
                    qh0, kh0, vh0, args.tile_cap, args.tile_cap
                )

        def fa2_fwd():
            with torch.no_grad():
                return flash_attn_func(q0, k0, v0, dropout_p=0.0, causal=True)

        local_fwd_ms = bench_cached(local_fwd, args.rep_ms)
        fa2_fwd_ms = bench_cached(fa2_fwd, args.rep_ms)
        fwd_config = config_string(local_attention._attention_forward_kernel.best_config)

        # Autograd CUDA Graph capture needs the leaves to originate on the
        # same non-default stream used for warmup, capture, and replay.
        train_stream = torch.cuda.Stream()
        with torch.cuda.stream(train_stream):
            q = torch.randn(
                batch, length, heads, dim,
                device="cuda", dtype=torch.bfloat16,
            ).requires_grad_()
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)
            grad = torch.randn_like(q)
            qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
            grad_h = grad.transpose(1, 2)
        train_stream.synchronize()

        def local_fwd_bwd():
            output = local_attention.block_causal_attention_triton(
                qh, kh, vh, args.tile_cap, args.tile_cap
            )
            grads = torch.autograd.grad(output, (q, k, v), grad_h)
            return output, grads

        def fa2_fwd_bwd():
            output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
            grads = torch.autograd.grad(output, (q, k, v), grad)
            return output, grads

        baseline_fb_ms = None
        if baseline is not None:
            def baseline_step():
                output = baseline.block_causal_attention_triton(
                    qh, kh, vh, args.tile_cap, args.tile_cap
                )
                return output, torch.autograd.grad(output, (q, k, v), grad_h)

            baseline_fb_ms = bench_fwd_bwd_cudagraph(
                baseline_step, train_stream, args.rep_ms
            )

        local_fb_ms = bench_fwd_bwd_cudagraph(
            local_fwd_bwd, train_stream, args.rep_ms
        )
        fa2_fb_ms = bench_fwd_bwd_cudagraph(
            fa2_fwd_bwd, train_stream, args.rep_ms
        )
        bwd_config = config_string(
            local_attention._attention_backward_kv_outer_kernel.best_config
        )

        with torch.cuda.stream(train_stream):
            local_output, local_grads = local_fwd_bwd()
            fa2_output, fa2_grads = fa2_fwd_bwd()
            torch.testing.assert_close(
                local_output.transpose(1, 2), fa2_output, rtol=3e-2, atol=3e-2
            )
            for actual, expected in zip(local_grads, fa2_grads):
                torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
            max_output_error = (local_output.transpose(1, 2) - fa2_output).abs().max().item()
            max_grad_error = max((a - b).abs().max().item() for a, b in zip(local_grads, fa2_grads))
        train_stream.synchronize()
        local_kernels = fa2_kernels = None
        if args.profile_kernels:
            local_kernels = profile_kernel_times(local_fwd_bwd, train_stream)
            fa2_kernels = profile_kernel_times(fa2_fwd_bwd, train_stream)
        records.append(dict(
            length=length, batch=batch, heads=heads, dim=dim, dtype="bfloat16",
            device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
            local_fwd_us=1000 * local_fwd_ms, fa2_fwd_us=1000 * fa2_fwd_ms,
            local_fwd_bwd_us=1000 * local_fb_ms, fa2_fwd_bwd_us=1000 * fa2_fb_ms,
            forward_config=fwd_config, backward_config=bwd_config,
            max_output_error=max_output_error, max_grad_error=max_grad_error,
            baseline_fwd_bwd_us=None if baseline_fb_ms is None else 1000 * baseline_fb_ms,
            local_kernels=local_kernels, fa2_kernels=fa2_kernels,
        ))

        print(
            f"{length}, {local_fwd_ms:.4f}, {fa2_fwd_ms:.4f}, "
            f"{local_fb_ms:.4f}, {fa2_fb_ms:.4f}, "
            f"{fwd_config}, {bwd_config}",
            flush=True,
        )
        if baseline_fb_ms is not None:
            print(f"  baseline fwd+bwd: {1000 * baseline_fb_ms:.2f} us", flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
