"""Causal attention with a fused Triton online-softmax forward pass.

The fused path keeps only Q, K, V, O and the per-query log-sum-exp (LSE) for
backward. It recomputes score/probability tiles in the backward pass instead
of saving the quadratic attention matrix. CPU, float64, and explicit-mask
inputs use :mod:`block_attention`'s PyTorch implementation.

Precision follows ``block_attention`` under AMP: FP16/BF16 operands use native
Tensor Core dots with float32 accumulation; FP32 inputs use the selected TF32
precision. Online-softmax state and gradient accumulators stay float32. Results
and gradients are written in the input dtype.
"""

import math

import torch
from torch.autograd.function import once_differentiable

from block_attention import block_causal_attention as _torch_block_attention

try:
    import triton
    import triton.language as tl
    import triton.testing
except ImportError:  # Keep the reference implementation usable without Triton.
    triton = None
    class _TritonLanguageFallback:
        # The annotations are evaluated when this module is imported, even
        # though the reference path never executes these kernel bodies.
        constexpr = object()

    tl = _TritonLanguageFallback()


_jit = triton.jit if triton is not None else (lambda function: function)

try:
    from triton.experimental import gluon as _gluon
    from triton.experimental.gluon import language as _gl

    _gluon_jit = _gluon.jit
except (ImportError, AttributeError):  # Gluon is optional on older Triton installs.
    _gluon = _gl = None
    _gluon_jit = lambda function: function


def _prune_tile_configs(configs, named_args, **kwargs):
    """Keep autotune tiles within the user's requested tile-size limits."""
    max_m = named_args["max_block_m"]
    max_n = named_args["max_block_n"]
    return [
        config for config in configs
        if config.kwargs["block_m"] <= max_m and config.kwargs["block_n"] <= max_n
    ]


def _attention_batch_layout(tensor):
    """Return flattened batch/head indexing and logical row/feature strides."""
    if tensor.ndim == 2:
        return (1, 1, 0, 0, tensor.stride(0), tensor.stride(1))
    if tensor.ndim == 3:
        return (
            tensor.shape[0], 1, tensor.stride(0), 0,
            tensor.stride(1), tensor.stride(2),
        )
    if tensor.ndim == 4:
        return (
            tensor.shape[0] * tensor.shape[1], tensor.shape[1],
            tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3),
        )
    raise ValueError("the Triton path supports rank-2, rank-3, or rank-4 Q/K/V")


if triton is not None:
    _autotune = triton.autotune
    _autotune_bench = triton.testing.do_bench_cudagraph
    _FORWARD_CONFIGS = [
        triton.Config({"block_m": 16, "block_n": 16}, num_warps=4, num_stages=1),
        triton.Config({"block_m": 32, "block_n": 32}, num_warps=4, num_stages=1),
        triton.Config({"block_m": 32, "block_n": 64}, num_warps=4, num_stages=2),
        triton.Config({"block_m": 64, "block_n": 32}, num_warps=4, num_stages=2),
        triton.Config({"block_m": 64, "block_n": 64}, num_warps=4, num_stages=2),
        triton.Config({"block_m": 128, "block_n": 32}, num_warps=4, num_stages=2),
        triton.Config({"block_m": 32, "block_n": 128}, num_warps=4, num_stages=2),
    ]
    _BACKWARD_CONFIGS = [
        triton.Config({"block_m": 16, "block_n": 16}, num_warps=4, num_stages=1),
        triton.Config({"block_m": 16, "block_n": 32}, num_warps=4, num_stages=1),
        triton.Config({"block_m": 32, "block_n": 16}, num_warps=4, num_stages=1),
        triton.Config({"block_m": 32, "block_n": 32}, num_warps=4, num_stages=1),
    ]
    _KV_OUTER_TILE_PAIRS = (
        (16, 16), (32, 32), (32, 64), (64, 32),
        (64, 64), (64, 128), (128, 64), (128, 128),
    )
    _KV_OUTER_CONFIGS = [
        triton.Config(
            {"block_m": block_m, "block_n": block_n},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_m, block_n in _KV_OUTER_TILE_PAIRS
        for num_warps, num_stages in ((4, 1), (4, 3), (8, 2))
    ]
    # Sweep MMA warp partitions at each tile. [4, 1] is the FA2 Sliced-Q
    # mapping; [1, 4] and [2, 2] are controls with identical kernel math.
    _SLICED_Q_CONFIGS = [
        triton.Config(
            {"block_m": block_m, "block_n": block_n, "warps_m": warps_m},
            num_warps=4,
            num_stages=1 if block_n <= 32 else 2,
        )
        for block_m in (64, 128)
        for block_n in (16, 32, 64, 128)
        for warps_m in (1, 2, 4)
    ]
else:
    _autotune = lambda **kwargs: (lambda function: function)
    _autotune_bench = None
    _FORWARD_CONFIGS = _BACKWARD_CONFIGS = []
    _KV_OUTER_CONFIGS = []
    _SLICED_Q_CONFIGS = []


@_autotune(
    configs=_FORWARD_CONFIGS,
    key=[
        "seq_q", "seq_k", "batch_size", "num_heads", "dim_qk", "dim_v",
        "max_block_m", "max_block_n", "precision_fp32", "input_dtype_code",
        "q_batch_stride", "q_head_stride", "q_seq_stride", "q_dim_stride",
        "k_batch_stride", "k_head_stride", "k_seq_stride", "k_dim_stride",
        "v_batch_stride", "v_head_stride", "v_seq_stride", "v_dim_stride",
    ],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    do_bench=_autotune_bench,
)
@_jit
def _attention_forward_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    seq_q, seq_k, batch_size, num_heads,
    dim_qk: tl.constexpr, dim_v: tl.constexpr,
    q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride,
    scale: tl.constexpr,
    precision_qk: tl.constexpr, precision_fp32: tl.constexpr,
    input_dtype_code,
    max_block_m, max_block_n,
    block_m: tl.constexpr, block_n: tl.constexpr,
    block_d: tl.constexpr, block_dv: tl.constexpr,
):
    pid_hb = tl.program_id(1)
    batch = pid_hb // num_heads
    head = pid_hb % num_heads
    pid_m = tl.program_id(0)

    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_n)
    dims = tl.arange(0, block_d)
    dims_v = tl.arange(0, block_dv)

    q = tl.load(
        q_ptr + batch * q_batch_stride + head * q_head_stride
        + rows[:, None] * q_seq_stride + dims[None, :] * q_dim_stride,
        (rows[:, None] < seq_q) & (dims[None, :] < dim_qk),
        other=0,
    )
    low_precision: tl.constexpr = q_ptr.dtype.element_ty == tl.float16 or q_ptr.dtype.element_ty == tl.bfloat16

    k_base = k_ptr + batch * k_batch_stride + head * k_head_stride
    v_base = v_ptr + batch * v_batch_stride + head * v_head_stride
    k_block_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(dim_qk, seq_k),
        strides=(k_dim_stride, k_seq_stride),
        offsets=(0, 0),
        block_shape=(block_d, block_n),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(dim_v, seq_k),
        strides=(v_dim_stride, v_seq_stride),
        offsets=(0, 0),
        block_shape=(block_dv, block_n),
        order=(0, 1),
    )

    row_max = tl.full((block_m,), -float("inf"), tl.float32)
    row_sum = tl.zeros((block_m,), tl.float32)
    out_acc = tl.zeros((block_m, block_dv), tl.float32)

    # Visit fully valid key tiles first. Only tiles crossing the causal
    # diagonal need the per-element comparison; increasing start_n naturally
    # leaves those tiles at the end of the traversal.
    end_n = tl.minimum((pid_m + 1) * block_m, seq_k)
    for start_n in range(0, end_n, block_n):
        key_cols = start_n + cols
        k = tl.trans(tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero"))
        v = tl.trans(tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero"))

        if low_precision:
            scores = tl.dot(q, tl.trans(k)) * scale
        else:
            scores = tl.dot(q, tl.trans(k), input_precision=precision_qk) * scale
        allowed = (rows[:, None] < seq_q) & (key_cols[None, :] < seq_k)
        if start_n + block_n > pid_m * block_m:
            allowed = allowed & (key_cols[None, :] <= rows[:, None])
        scores = tl.where(allowed, scores, -float("inf"))

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        old_scale = tl.exp(row_max - safe_max)
        probs = tl.exp(scores - safe_max[:, None])

        if low_precision:
            pv = tl.dot(probs.to(q_ptr.dtype.element_ty), v)
        elif precision_fp32 == "bf16":
            pv = tl.dot(probs.to(tl.bfloat16), v.to(tl.bfloat16))
        else:
            pv = tl.dot(probs, v, input_precision=precision_fp32)
        out_acc = out_acc * old_scale[:, None] + pv
        row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
        row_max = new_max
        k_block_ptr = tl.advance(k_block_ptr, (0, block_n))
        v_block_ptr = tl.advance(v_block_ptr, (0, block_n))

    safe_sum = tl.where(row_sum > 0.0, row_sum, 1.0)
    output = out_acc / safe_sum[:, None]
    lse = tl.where(row_sum > 0.0, row_max + tl.log(safe_sum), -float("inf"))

    tl.store(
        o_ptr + pid_hb * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
        output,
        (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v),
    )
    tl.store(lse_ptr + pid_hb * seq_q + rows, lse, rows < seq_q)


if _gluon is not None:
    @_autotune(
        configs=_SLICED_Q_CONFIGS,
        key=[
            "seq_q", "seq_k", "batch_size", "dim_qk", "dim_v",
            "max_block_m", "max_block_n", "precision_fp32", "input_dtype_code",
        ],
        prune_configs_by={"early_config_prune": _prune_tile_configs},
        do_bench=_autotune_bench,
    )
    @_gluon_jit
    def _attention_forward_sliced_q_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
        seq_q, seq_k, batch_size, dim_qk: _gl.constexpr, dim_v: _gl.constexpr,
        scale: _gl.constexpr,
        precision_fp32: _gl.constexpr,
        input_dtype_code,
        max_block_m, max_block_n,
        warps_m: _gl.constexpr,
        block_m: _gl.constexpr, block_n: _gl.constexpr,
        block_d: _gl.constexpr, block_dv: _gl.constexpr,
    ):
        """Online-softmax forward using an explicit FA2-style Sliced-Q MMA layout."""
        batch = _gl.program_id(1)
        pid_m = _gl.program_id(0)

        # Load Q with a row-oriented layout; MMA repartitions it according
        # to warps_m. K/V are converted to the selected dot-operand layout.
        q_layout: _gl.constexpr = _gl.BlockedLayout(
            [1, 4], [4, 8], [4, 1], [1, 0]
        )
        kv_layout: _gl.constexpr = _gl.BlockedLayout(
            [1, 4], [4, 8], [1, 4], [1, 0]
        )
        mma_layout: _gl.constexpr = _gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[warps_m, 4 // warps_m],
            instr_shape=[16, 8],
        )
        q_rows = pid_m * block_m + _gl.arange(
            0, block_m, layout=_gl.SliceLayout(1, q_layout)
        )
        q_dims = _gl.arange(0, block_d, layout=_gl.SliceLayout(0, q_layout))
        kv_rows = _gl.arange(0, block_n, layout=_gl.SliceLayout(1, kv_layout))
        kv_dims = _gl.arange(0, block_d, layout=_gl.SliceLayout(0, kv_layout))
        v_dims = _gl.arange(0, block_dv, layout=_gl.SliceLayout(0, kv_layout))

        q = _gl.load(
            q_ptr + batch * seq_q * dim_qk + q_rows[:, None] * dim_qk + q_dims[None, :],
            (q_rows[:, None] < seq_q) & (q_dims[None, :] < dim_qk),
            other=0,
        )
        row_layout: _gl.constexpr = _gl.SliceLayout(1, mma_layout)
        col_layout: _gl.constexpr = _gl.SliceLayout(0, mma_layout)
        rows = pid_m * block_m + _gl.arange(0, block_m, layout=row_layout)
        cols = _gl.arange(0, block_n, layout=col_layout)
        qk_width: _gl.constexpr = 1 if q_ptr.dtype.element_ty == _gl.float32 else 2
        q_operand = _gl.convert_layout(
            q, _gl.DotOperandLayout(0, mma_layout, k_width=qk_width)
        )

        row_max = _gl.full((block_m,), -float("inf"), _gl.float32, layout=row_layout)
        row_sum = _gl.zeros((block_m,), _gl.float32, layout=row_layout)
        out_acc = _gl.zeros((block_m, block_dv), _gl.float32, layout=mma_layout)

        end_n = _gl.minimum((pid_m + 1) * block_m, seq_k)
        for start_n in range(0, end_n, block_n):
            key_rows = start_n + kv_rows
            key_cols = start_n + cols
            k = _gl.load(
                k_ptr + batch * seq_k * dim_qk + key_rows[:, None] * dim_qk + kv_dims[None, :],
                (key_rows[:, None] < seq_k) & (kv_dims[None, :] < dim_qk),
                other=0,
            )
            v = _gl.load(
                v_ptr + batch * seq_k * dim_v + key_rows[:, None] * dim_v + v_dims[None, :],
                (key_rows[:, None] < seq_k) & (v_dims[None, :] < dim_v),
                other=0,
            )
            kt_operand = _gl.convert_layout(
                _gl.permute(k, (1, 0)),
                _gl.DotOperandLayout(1, mma_layout, k_width=qk_width),
            )
            scores = _gl.nvidia.ampere.mma_v2(
                q_operand,
                kt_operand,
                _gl.zeros((block_m, block_n), _gl.float32, layout=mma_layout),
                input_precision="tf32" if qk_width == 1 else None,
            ) * scale
            allowed = (rows[:, None] < seq_q) & (key_cols[None, :] < seq_k)
            if start_n + block_n > pid_m * block_m:
                allowed = allowed & (key_cols[None, :] <= rows[:, None])
            scores = _gl.where(allowed, scores, -float("inf"))

            tile_max = _gl.max(scores, 1)
            new_max = _gl.maximum(row_max, tile_max)
            safe_max = _gl.where(new_max == -float("inf"), 0.0, new_max)
            old_scale = _gl.exp(row_max - safe_max)
            probs = _gl.exp(scores - safe_max[:, None])

            if qk_width == 2:
                pv_width: _gl.constexpr = 2
                p_operand = _gl.convert_layout(
                    probs.to(q_ptr.dtype.element_ty),
                    _gl.DotOperandLayout(0, mma_layout, k_width=pv_width),
                )
                v_operand = _gl.convert_layout(
                    v, _gl.DotOperandLayout(1, mma_layout, k_width=pv_width)
                )
                pv = _gl.nvidia.ampere.mma_v2(
                    p_operand,
                    v_operand,
                    _gl.zeros((block_m, block_dv), _gl.float32, layout=mma_layout),
                )
            elif precision_fp32 == "bf16":
                pv_width: _gl.constexpr = 2
                p_operand = _gl.convert_layout(
                    probs.to(_gl.bfloat16),
                    _gl.DotOperandLayout(0, mma_layout, k_width=pv_width),
                )
                v_operand = _gl.convert_layout(
                    v.to(_gl.bfloat16),
                    _gl.DotOperandLayout(1, mma_layout, k_width=pv_width),
                )
                pv = _gl.nvidia.ampere.mma_v2(
                    p_operand,
                    v_operand,
                    _gl.zeros((block_m, block_dv), _gl.float32, layout=mma_layout),
                )
            else:
                p_operand = _gl.convert_layout(
                    probs,
                    _gl.DotOperandLayout(0, mma_layout, k_width=1),
                )
                v_operand = _gl.convert_layout(
                    v.to(_gl.float32),
                    _gl.DotOperandLayout(1, mma_layout, k_width=1),
                )
                if precision_fp32 == "tf32x3":
                    # Gluon's explicit mma_v2 currently lowers one TF32 dot
                    # at a time, so reproduce Triton's 3x-TF32 decomposition.
                    p_hi = _gl.inline_asm_elementwise(
                        "cvt.rna.tf32.f32 $0, $1;", "=r,r", [probs],
                        dtype=_gl.float32, is_pure=True, pack=1,
                    )
                    v_hi = _gl.inline_asm_elementwise(
                        "cvt.rna.tf32.f32 $0, $1;", "=r,r", [v.to(_gl.float32)],
                        dtype=_gl.float32, is_pure=True, pack=1,
                    )
                    p_lo, v_lo = probs - p_hi, v.to(_gl.float32) - v_hi
                    p_hi_operand = _gl.convert_layout(
                        p_hi, _gl.DotOperandLayout(0, mma_layout, k_width=1)
                    )
                    p_lo_operand = _gl.convert_layout(
                        p_lo, _gl.DotOperandLayout(0, mma_layout, k_width=1)
                    )
                    v_hi_operand = _gl.convert_layout(
                        v_hi, _gl.DotOperandLayout(1, mma_layout, k_width=1)
                    )
                    v_lo_operand = _gl.convert_layout(
                        v_lo, _gl.DotOperandLayout(1, mma_layout, k_width=1)
                    )
                    pv = _gl.nvidia.ampere.mma_v2(
                        p_hi_operand, v_hi_operand,
                        _gl.zeros((block_m, block_dv), _gl.float32, layout=mma_layout),
                        input_precision="tf32",
                    )
                    pv = _gl.nvidia.ampere.mma_v2(
                        p_lo_operand, v_hi_operand, pv, input_precision="tf32"
                    )
                    pv = _gl.nvidia.ampere.mma_v2(
                        p_hi_operand, v_lo_operand, pv, input_precision="tf32"
                    )
                else:
                    pv = _gl.nvidia.ampere.mma_v2(
                        p_operand,
                        v_operand,
                        _gl.zeros((block_m, block_dv), _gl.float32, layout=mma_layout),
                        input_precision=precision_fp32,
                    )
            out_acc = out_acc * old_scale[:, None] + pv
            row_sum = row_sum * old_scale + _gl.sum(probs, 1)
            row_max = new_max

        safe_sum = _gl.where(row_sum > 0.0, row_sum, 1.0)
        output = out_acc / safe_sum[:, None]
        lse = _gl.where(row_sum > 0.0, row_max + _gl.log(safe_sum), -float("inf"))
        out_rows = pid_m * block_m + _gl.arange(0, block_m, layout=row_layout)
        out_dims = _gl.arange(0, block_dv, layout=col_layout)
        _gl.store(
            o_ptr + batch * seq_q * dim_v + out_rows[:, None] * dim_v + out_dims[None, :],
            output,
            (out_rows[:, None] < seq_q) & (out_dims[None, :] < dim_v),
        )
        _gl.store(lse_ptr + batch * seq_q + out_rows, lse, out_rows < seq_q)
else:
    _attention_forward_sliced_q_kernel = None


@_jit
def _attention_backward_tile(
    q, k, v, do, safe_lse, delta, rows, key_cols,
    seq_q, seq_k, scale: tl.constexpr,
    precision_qk: tl.constexpr, precision_fp32: tl.constexpr,
    precision_do_v: tl.constexpr, input_dtype: tl.constexpr,
    diagonal_tile,
):
    """Recompute one Q/KV tile and return its dQ, dK, and dV contributions."""
    low_precision: tl.constexpr = input_dtype == tl.float16 or input_dtype == tl.bfloat16
    do_native = do.to(input_dtype)
    if low_precision:
        scores = tl.dot(q, tl.trans(k)) * scale
        dp = tl.dot(do_native, tl.trans(v))
    else:
        scores = tl.dot(q, tl.trans(k), input_precision=precision_qk) * scale
        dp = tl.dot(do, tl.trans(v), input_precision=precision_do_v)

    allowed = (rows[:, None] < seq_q) & (key_cols[None, :] < seq_k)
    if diagonal_tile:
        allowed = allowed & (key_cols[None, :] <= rows[:, None])
    scores = tl.where(allowed, scores, -float("inf"))
    probs = tl.exp(scores - safe_lse[:, None])
    ds = probs * (dp - delta[:, None])

    if low_precision:
        ds_native = ds.to(input_dtype)
        probs_native = probs.to(input_dtype)
        dq = tl.dot(ds_native, k) * scale
        dk = tl.dot(tl.trans(ds_native), q) * scale
        dv = tl.dot(tl.trans(probs_native), do_native)
    elif precision_fp32 == "bf16":
        ds_bf16 = ds.to(tl.bfloat16)
        probs_bf16 = probs.to(tl.bfloat16)
        q_bf16 = q.to(tl.bfloat16)
        k_bf16 = k.to(tl.bfloat16)
        do_bf16 = do.to(tl.bfloat16)
        dq = tl.dot(ds_bf16, k_bf16) * scale
        dk = tl.dot(tl.trans(ds_bf16), q_bf16) * scale
        dv = tl.dot(tl.trans(probs_bf16), do_bf16)
    else:
        dq = tl.dot(ds, k, input_precision=precision_fp32) * scale
        dk = tl.dot(tl.trans(ds), q, input_precision=precision_fp32) * scale
        dv = tl.dot(tl.trans(probs), do, input_precision=precision_fp32)
    return dq, dk, dv


@_autotune(
    configs=_BACKWARD_CONFIGS,
    key=["seq_q", "seq_k", "batch_size", "dim_qk", "dim_v", "max_block_m", "max_block_n", "precision_fp32", "input_dtype_code"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    do_bench=_autotune_bench,
)
@_jit
def _attention_backward_dq_owner_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, do_ptr, dq_ptr,
    seq_q, seq_k, batch_size,
    dim_qk: tl.constexpr, dim_v: tl.constexpr,
    scale: tl.constexpr,
    precision_qk: tl.constexpr, precision_fp32: tl.constexpr,
    precision_do_v: tl.constexpr, input_dtype_code,
    max_block_m, max_block_n,
    block_m: tl.constexpr, block_n: tl.constexpr,
    block_d: tl.constexpr, block_dv: tl.constexpr,
):
    """A CTA owns Q rows, traverses KV tiles, and writes its dQ tile once."""
    batch = tl.program_id(1)
    pid_m = tl.program_id(0)
    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_n)
    dims = tl.arange(0, block_d)
    dims_v = tl.arange(0, block_dv)
    q = tl.load(
        q_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
        (rows[:, None] < seq_q) & (dims[None, :] < dim_qk), other=0,
    )
    do = tl.load(
        do_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
        (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
    )
    o = tl.load(
        o_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
        (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
    ).to(tl.float32)
    lse = tl.load(lse_ptr + batch * seq_q + rows, rows < seq_q, other=-float("inf"))
    safe_lse = tl.where(lse == -float("inf"), 0.0, lse)
    delta = tl.sum(do.to(tl.float32) * o, axis=1)
    dq_acc = tl.zeros((block_m, block_d), tl.float32)

    q_start = pid_m * block_m
    end_n = tl.minimum(q_start + block_m, seq_k)
    for start_n in range(0, end_n, block_n):
        key_cols = start_n + cols
        k = tl.load(
            k_ptr + batch * seq_k * dim_qk + key_cols[:, None] * dim_qk + dims[None, :],
            (key_cols[:, None] < seq_k) & (dims[None, :] < dim_qk), other=0,
        )
        v = tl.load(
            v_ptr + batch * seq_k * dim_v + key_cols[:, None] * dim_v + dims_v[None, :],
            (key_cols[:, None] < seq_k) & (dims_v[None, :] < dim_v), other=0,
        )
        diagonal_tile = start_n + block_n > q_start
        dq_tile, _, _ = _attention_backward_tile(
            q, k, v, do, safe_lse, delta, rows, key_cols,
            seq_q, seq_k, scale, precision_qk, precision_fp32,
            precision_do_v, q_ptr.dtype.element_ty, diagonal_tile,
        )
        dq_acc += dq_tile

    tl.store(
        dq_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
        dq_acc,
        (rows[:, None] < seq_q) & (dims[None, :] < dim_qk),
    )


@_autotune(
    configs=_BACKWARD_CONFIGS,
    key=["seq_q", "seq_k", "batch_size", "dim_qk", "dim_v", "max_block_m", "max_block_n", "precision_fp32", "input_dtype_code"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    do_bench=_autotune_bench,
)
@_jit
def _attention_backward_dkv_owner_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, do_ptr, dk_ptr, dv_ptr,
    seq_q, seq_k, batch_size,
    dim_qk: tl.constexpr, dim_v: tl.constexpr,
    scale: tl.constexpr,
    precision_qk: tl.constexpr, precision_fp32: tl.constexpr,
    precision_do_v: tl.constexpr, input_dtype_code,
    max_block_m, max_block_n,
    block_m: tl.constexpr, block_n: tl.constexpr,
    block_d: tl.constexpr, block_dv: tl.constexpr,
):
    """A CTA owns one KV tile, traverses Q tiles, and writes dK/dV once."""
    batch = tl.program_id(1)
    pid_n = tl.program_id(0)
    rows_k = pid_n * block_n + tl.arange(0, block_n)
    rows_m = tl.arange(0, block_m)
    dims = tl.arange(0, block_d)
    dims_v = tl.arange(0, block_dv)
    k = tl.load(
        k_ptr + batch * seq_k * dim_qk + rows_k[:, None] * dim_qk + dims[None, :],
        (rows_k[:, None] < seq_k) & (dims[None, :] < dim_qk), other=0,
    )
    v = tl.load(
        v_ptr + batch * seq_k * dim_v + rows_k[:, None] * dim_v + dims_v[None, :],
        (rows_k[:, None] < seq_k) & (dims_v[None, :] < dim_v), other=0,
    )
    key_start = pid_n * block_n
    key_end = tl.minimum(key_start + block_n, seq_k)
    num_query_tiles = tl.cdiv(seq_q, block_m)
    first_full_q = tl.cdiv(key_end, block_m)
    mixed_q_begin = key_start // block_m
    mixed_q_end = tl.minimum(tl.cdiv(key_end, block_m), num_query_tiles)
    dk_acc = tl.zeros((block_n, block_d), tl.float32)
    dv_acc = tl.zeros((block_n, block_dv), tl.float32)

    # Fully valid Q tiles come first; diagonal-intersecting tiles are handled
    # afterward with the elementwise causal mask.
    for pid_m in range(first_full_q, num_query_tiles):
        rows = pid_m * block_m + rows_m
        q = tl.load(
            q_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
            (rows[:, None] < seq_q) & (dims[None, :] < dim_qk), other=0,
        )
        do = tl.load(
            do_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
            (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
        )
        o = tl.load(
            o_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
            (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
        ).to(tl.float32)
        lse = tl.load(lse_ptr + batch * seq_q + rows, rows < seq_q, other=-float("inf"))
        safe_lse = tl.where(lse == -float("inf"), 0.0, lse)
        delta = tl.sum(do.to(tl.float32) * o, axis=1)
        _, dk_part, dv_part = _attention_backward_tile(
            q, k, v, do, safe_lse, delta, rows, rows_k,
            seq_q, seq_k, scale, precision_qk, precision_fp32,
            precision_do_v, q_ptr.dtype.element_ty, False,
        )
        dk_acc += dk_part
        dv_acc += dv_part

    for pid_m in range(mixed_q_begin, mixed_q_end):
        rows = pid_m * block_m + rows_m
        q = tl.load(
            q_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
            (rows[:, None] < seq_q) & (dims[None, :] < dim_qk), other=0,
        )
        do = tl.load(
            do_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
            (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
        )
        o = tl.load(
            o_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
            (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
        ).to(tl.float32)
        lse = tl.load(lse_ptr + batch * seq_q + rows, rows < seq_q, other=-float("inf"))
        safe_lse = tl.where(lse == -float("inf"), 0.0, lse)
        delta = tl.sum(do.to(tl.float32) * o, axis=1)
        _, dk_part, dv_part = _attention_backward_tile(
            q, k, v, do, safe_lse, delta, rows, rows_k,
            seq_q, seq_k, scale, precision_qk, precision_fp32,
            precision_do_v, q_ptr.dtype.element_ty, True,
        )
        dk_acc += dk_part
        dv_acc += dv_part

    tl.store(
        dk_ptr + batch * seq_k * dim_qk + rows_k[:, None] * dim_qk + dims[None, :],
        dk_acc, (rows_k[:, None] < seq_k) & (dims[None, :] < dim_qk),
    )
    tl.store(
        dv_ptr + batch * seq_k * dim_v + rows_k[:, None] * dim_v + dims_v[None, :],
        dv_acc, (rows_k[:, None] < seq_k) & (dims_v[None, :] < dim_v),
    )


@_autotune(
    configs=_BACKWARD_CONFIGS,
    key=["seq_q", "seq_k", "batch_size", "dim_qk", "dim_v", "max_block_m", "max_block_n", "precision_fp32", "input_dtype_code"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    do_bench=_autotune_bench,
)
@_jit
def _attention_backward_owner_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, do_ptr, dq_ptr, dk_ptr, dv_ptr,
    seq_q, seq_k, batch_size,
    dim_qk: tl.constexpr, dim_v: tl.constexpr,
    scale: tl.constexpr,
    precision_qk: tl.constexpr, precision_fp32: tl.constexpr,
    precision_do_v: tl.constexpr, input_dtype_code,
    max_block_m, max_block_n,
    block_m: tl.constexpr, block_n: tl.constexpr,
    block_d: tl.constexpr, block_dv: tl.constexpr,
):
    """One launch with Q-owned and KV-owned CTAs writing disjoint outputs."""
    batch = tl.program_id(1)
    pid = tl.program_id(0)
    num_query_tiles = tl.cdiv(seq_q, block_m)
    rows_m = tl.arange(0, block_m)
    cols_n = tl.arange(0, block_n)
    dims = tl.arange(0, block_d)
    dims_v = tl.arange(0, block_dv)

    # The first CTA range owns Q tiles and reduces across all causally valid KV
    # tiles. Fully valid tiles precede the diagonal tile and need no comparison.
    if pid < num_query_tiles:
        rows = pid * block_m + rows_m
        q = tl.load(
            q_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
            (rows[:, None] < seq_q) & (dims[None, :] < dim_qk), other=0,
        )
        do = tl.load(
            do_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
            (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
        )
        o = tl.load(
            o_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
            (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
        ).to(tl.float32)
        lse = tl.load(lse_ptr + batch * seq_q + rows, rows < seq_q, other=-float("inf"))
        safe_lse = tl.where(lse == -float("inf"), 0.0, lse)
        delta = tl.sum(do.to(tl.float32) * o, axis=1)
        dq_acc = tl.zeros((block_m, block_d), tl.float32)
        q_start = pid * block_m
        end_n = tl.minimum(q_start + block_m, seq_k)
        for start_n in range(0, end_n, block_n):
            key_cols = start_n + cols_n
            k = tl.load(
                k_ptr + batch * seq_k * dim_qk + key_cols[:, None] * dim_qk + dims[None, :],
                (key_cols[:, None] < seq_k) & (dims[None, :] < dim_qk), other=0,
            )
            v = tl.load(
                v_ptr + batch * seq_k * dim_v + key_cols[:, None] * dim_v + dims_v[None, :],
                (key_cols[:, None] < seq_k) & (dims_v[None, :] < dim_v), other=0,
            )
            diagonal_tile = start_n + block_n > q_start
            dq_tile, _, _ = _attention_backward_tile(
                q, k, v, do, safe_lse, delta, rows, key_cols,
                seq_q, seq_k, scale, precision_qk, precision_fp32,
                precision_do_v, q_ptr.dtype.element_ty, diagonal_tile,
            )
            dq_acc += dq_tile
        tl.store(
            dq_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
            dq_acc, (rows[:, None] < seq_q) & (dims[None, :] < dim_qk),
        )
    else:
        # The remaining CTAs own KV tiles and reduce across Q. Fully valid Q
        # tiles run first; only diagonal-intersecting tiles apply causal masks.
        pid_n = pid - num_query_tiles
        rows_k = pid_n * block_n + cols_n
        k = tl.load(
            k_ptr + batch * seq_k * dim_qk + rows_k[:, None] * dim_qk + dims[None, :],
            (rows_k[:, None] < seq_k) & (dims[None, :] < dim_qk), other=0,
        )
        v = tl.load(
            v_ptr + batch * seq_k * dim_v + rows_k[:, None] * dim_v + dims_v[None, :],
            (rows_k[:, None] < seq_k) & (dims_v[None, :] < dim_v), other=0,
        )
        key_start = pid_n * block_n
        key_end = tl.minimum(key_start + block_n, seq_k)
        first_full_q = tl.cdiv(key_end, block_m)
        mixed_q_begin = key_start // block_m
        mixed_q_end = tl.minimum(tl.cdiv(key_end, block_m), num_query_tiles)
        dk_acc = tl.zeros((block_n, block_d), tl.float32)
        dv_acc = tl.zeros((block_n, block_dv), tl.float32)

        for pid_m in range(first_full_q, num_query_tiles):
            rows = pid_m * block_m + rows_m
            q = tl.load(
                q_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
                (rows[:, None] < seq_q) & (dims[None, :] < dim_qk), other=0,
            )
            do = tl.load(
                do_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
                (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
            )
            o = tl.load(
                o_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
                (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
            ).to(tl.float32)
            lse = tl.load(lse_ptr + batch * seq_q + rows, rows < seq_q, other=-float("inf"))
            safe_lse = tl.where(lse == -float("inf"), 0.0, lse)
            delta = tl.sum(do.to(tl.float32) * o, axis=1)
            _, dk_part, dv_part = _attention_backward_tile(
                q, k, v, do, safe_lse, delta, rows, rows_k,
                seq_q, seq_k, scale, precision_qk, precision_fp32,
                precision_do_v, q_ptr.dtype.element_ty, False,
            )
            dk_acc += dk_part
            dv_acc += dv_part

        for pid_m in range(mixed_q_begin, mixed_q_end):
            rows = pid_m * block_m + rows_m
            q = tl.load(
                q_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
                (rows[:, None] < seq_q) & (dims[None, :] < dim_qk), other=0,
            )
            do = tl.load(
                do_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
                (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
            )
            o = tl.load(
                o_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
                (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v), other=0,
            ).to(tl.float32)
            lse = tl.load(lse_ptr + batch * seq_q + rows, rows < seq_q, other=-float("inf"))
            safe_lse = tl.where(lse == -float("inf"), 0.0, lse)
            delta = tl.sum(do.to(tl.float32) * o, axis=1)
            _, dk_part, dv_part = _attention_backward_tile(
                q, k, v, do, safe_lse, delta, rows, rows_k,
                seq_q, seq_k, scale, precision_qk, precision_fp32,
                precision_do_v, q_ptr.dtype.element_ty, True,
            )
            dk_acc += dk_part
            dv_acc += dv_part

        tl.store(
            dk_ptr + batch * seq_k * dim_qk + rows_k[:, None] * dim_qk + dims[None, :],
            dk_acc, (rows_k[:, None] < seq_k) & (dims[None, :] < dim_qk),
        )
        tl.store(
            dv_ptr + batch * seq_k * dim_v + rows_k[:, None] * dim_v + dims_v[None, :],
            dv_acc, (rows_k[:, None] < seq_k) & (dims_v[None, :] < dim_v),
        )


@_jit
def _attention_delta_kernel(
    o_ptr, do_ptr, delta_ptr,
    seq_q, dim_v, batch_size,
    do_batch_stride, do_seq_stride, do_dim_stride,
    block_m: tl.constexpr, block_dv: tl.constexpr,
):
    """Compute delta[b, i] = sum_d(dO[b, i, d] * O[b, i, d]) in FP32."""
    pid_hb = tl.program_id(1)
    pid_m = tl.program_id(0)
    rows = pid_m * block_m + tl.arange(0, block_m)
    dims_v = tl.arange(0, block_dv)
    do = tl.load(
        do_ptr + pid_hb * do_batch_stride
        + rows[:, None] * do_seq_stride + dims_v[None, :] * do_dim_stride,
        (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v),
        other=0,
    ).to(tl.float32)
    o = tl.load(
        o_ptr + pid_hb * seq_q * dim_v
        + rows[:, None] * dim_v + dims_v[None, :],
        (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v),
        other=0,
    ).to(tl.float32)
    delta = tl.sum(do * o, axis=1)
    tl.store(delta_ptr + pid_hb * seq_q + rows, delta, rows < seq_q)


@_jit
def _cast_fp32_gradient_kernel(
    src_ptr, dst_ptr, numel, block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    values = tl.load(src_ptr + offsets, offsets < numel, other=0.0)
    tl.store(dst_ptr + offsets, values, offsets < numel)


@_autotune(
    configs=_KV_OUTER_CONFIGS,
    key=[
        "seq_q", "seq_k", "batch_size", "num_heads", "dim_qk", "dim_v",
        "max_block_m", "max_block_n", "precision_fp32", "input_dtype_code",
        "q_batch_stride", "q_head_stride", "q_seq_stride", "q_dim_stride",
        "k_batch_stride", "k_head_stride", "k_seq_stride", "k_dim_stride",
        "v_batch_stride", "v_head_stride", "v_seq_stride", "v_dim_stride",
    ],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    do_bench=_autotune_bench,
    reset_to_zero=["dq_acc_ptr"],
)
@_jit
def _attention_backward_kv_outer_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, do_ptr, delta_ptr,
    dq_acc_ptr, dk_ptr, dv_ptr,
    seq_q, seq_k, batch_size, num_heads,
    dim_qk: tl.constexpr, dim_v: tl.constexpr,
    q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
    k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
    v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride,
    scale: tl.constexpr,
    precision_qk: tl.constexpr, precision_fp32: tl.constexpr,
    precision_do_v: tl.constexpr, input_dtype_code,
    do_batch_stride, do_seq_stride, do_dim_stride,
    max_block_m, max_block_n,
    block_m: tl.constexpr, block_n: tl.constexpr,
    block_d: tl.constexpr, block_dv: tl.constexpr,
):
    """One CTA owns a KV tile, accumulates dK/dV, and atomically adds dQ."""
    pid_hb = tl.program_id(1)
    batch = pid_hb // num_heads
    head = pid_hb % num_heads
    pid_n = tl.program_id(0)
    rows_k = pid_n * block_n + tl.arange(0, block_n)
    cols = tl.arange(0, block_n)
    rows_m = tl.arange(0, block_m)
    dims = tl.arange(0, block_d)
    dims_v = tl.arange(0, block_dv)
    key_start = pid_n * block_n
    key_end = tl.minimum(key_start + block_n, seq_k)

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + batch * k_batch_stride + head * k_head_stride,
        shape=(dim_qk, seq_k),
        strides=(k_dim_stride, k_seq_stride),
        offsets=(0, key_start),
        block_shape=(block_d, block_n),
        order=(0, 1),
    )
    k = tl.trans(tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero"))
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + batch * v_batch_stride + head * v_head_stride,
        shape=(seq_k, dim_v),
        strides=(v_seq_stride, v_dim_stride),
        offsets=(key_start, 0),
        block_shape=(block_n, block_dv),
        order=(1, 0),
    )
    v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

    low_precision: tl.constexpr = (
        q_ptr.dtype.element_ty == tl.float16
        or q_ptr.dtype.element_ty == tl.bfloat16
    )
    if low_precision:
        q_dtype: tl.constexpr = q_ptr.dtype.element_ty
        k_scaled = (k.to(tl.float32) * scale).to(q_dtype)
    else:
        k_scaled = k.to(tl.float32) * scale

    num_query_tiles = tl.cdiv(seq_q, block_m)
    first_full_q = tl.cdiv(key_end, block_m)
    first_boundary_q = key_start // block_m
    dk_acc = tl.zeros((block_n, block_d), tl.float32)
    dv_acc = tl.zeros((block_n, block_dv), tl.float32)

    # Reverse Q traversal visits all full causal tiles before the diagonal and
    # boundary tiles. Fully valid tiles do not perform an elementwise compare.
    for reverse_m in range(0, num_query_tiles):
        pid_m = num_query_tiles - 1 - reverse_m
        if pid_m >= first_full_q:
            diagonal_tile = False
        elif pid_m >= first_boundary_q:
            diagonal_tile = True
        else:
            diagonal_tile = False

        if (pid_m >= first_full_q) | (pid_m >= first_boundary_q):
            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + batch * q_batch_stride + head * q_head_stride,
                shape=(seq_q, dim_qk),
                strides=(q_seq_stride, q_dim_stride),
                offsets=(pid_m * block_m, 0),
                block_shape=(block_m, block_d),
                order=(1, 0),
            )
            rows = pid_m * block_m + rows_m
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            do = tl.load(
                do_ptr + pid_hb * do_batch_stride
                + rows[:, None] * do_seq_stride + dims_v[None, :] * do_dim_stride,
                (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v),
                other=0,
            )
            o = tl.load(
                o_ptr + pid_hb * seq_q * dim_v
                + rows[:, None] * dim_v + dims_v[None, :],
                (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v),
                other=0,
            )
            lse = tl.load(lse_ptr + pid_hb * seq_q + rows, rows < seq_q, other=-float("inf"))
            delta = tl.load(delta_ptr + pid_hb * seq_q + rows, rows < seq_q, other=0.0)
            safe_lse_log2 = tl.where(lse == -float("inf"), 0.0, lse * 1.4426950408889634)

            if low_precision:
                q_scale = (q.to(tl.float32) * scale).to(q_ptr.dtype.element_ty)
                q_log2 = (q.to(tl.float32) * (scale * 1.4426950408889634)).to(q_ptr.dtype.element_ty)
                do_native = do.to(q_ptr.dtype.element_ty)
                scores = tl.dot(q_log2, tl.trans(k))
                dp = tl.dot(do_native, tl.trans(v))
            elif precision_fp32 == "bf16":
                q_scale = (q * scale).to(tl.bfloat16)
                q_log2 = (q * (scale * 1.4426950408889634)).to(tl.bfloat16)
                do_bf16 = do.to(tl.bfloat16)
                scores = tl.dot(q_log2, tl.trans(k.to(tl.bfloat16)))
                dp = tl.dot(do_bf16, tl.trans(v.to(tl.bfloat16)))
            else:
                q_scale = q.to(tl.float32) * scale
                q_log2 = q.to(tl.float32) * (scale * 1.4426950408889634)
                scores = tl.dot(q_log2, tl.trans(k), input_precision=precision_qk)
                dp = tl.dot(do, tl.trans(v), input_precision=precision_do_v)

            key_cols = rows_k
            allowed = (rows[:, None] < seq_q) & (key_cols[None, :] < seq_k)
            if diagonal_tile:
                allowed = allowed & (key_cols[None, :] <= rows[:, None])
            probs = tl.exp2(tl.where(allowed, scores - safe_lse_log2[:, None], -float("inf")))
            ds = probs * (dp.to(tl.float32) - delta[:, None])

            if low_precision:
                ds_operand = ds.to(q_ptr.dtype.element_ty)
                probs_operand = probs.to(q_ptr.dtype.element_ty)
                dq_tile = tl.dot(ds_operand, k_scaled)
                dk_part = tl.dot(tl.trans(ds_operand), q_scale)
                dv_part = tl.dot(tl.trans(probs_operand), do_native)
            elif precision_fp32 == "bf16":
                ds_bf16 = ds.to(tl.bfloat16)
                probs_bf16 = probs.to(tl.bfloat16)
                dq_tile = tl.dot(ds_bf16, k_scaled.to(tl.bfloat16))
                dk_part = tl.dot(tl.trans(ds_bf16), q_scale)
                dv_part = tl.dot(tl.trans(probs_bf16), do_bf16)
            else:
                dq_tile = tl.dot(ds, k_scaled, input_precision=precision_fp32)
                dk_part = tl.dot(tl.trans(ds), q_scale, input_precision=precision_fp32)
                dv_part = tl.dot(tl.trans(probs), do, input_precision=precision_fp32)

            q_offset = pid_hb * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :]
            tl.atomic_add(
                dq_acc_ptr + q_offset,
                dq_tile,
                mask=(rows[:, None] < seq_q) & (dims[None, :] < dim_qk),
                sem="relaxed",
            )
            dk_acc += dk_part
            dv_acc += dv_part

    tl.store(
        dk_ptr + pid_hb * seq_k * dim_qk + rows_k[:, None] * dim_qk + dims[None, :],
        dk_acc, (rows_k[:, None] < seq_k) & (dims[None, :] < dim_qk),
    )
    tl.store(
        dv_ptr + pid_hb * seq_k * dim_v + rows_k[:, None] * dim_v + dims_v[None, :],
        dv_acc, (rows_k[:, None] < seq_k) & (dims_v[None, :] < dim_v),
    )


@_autotune(
    configs=_BACKWARD_CONFIGS,
    key=["seq_q", "seq_k", "batch_size", "dim_qk", "dim_v", "max_block_m", "max_block_n", "precision_fp32", "input_dtype_code", "use_atomic"],
    prune_configs_by={"early_config_prune": _prune_tile_configs},
    reset_to_zero=["dkv_acc_ptr"],
)
@_jit
def _attention_backward_fused_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, do_ptr, dq_ptr, dkv_acc_ptr, dkv_partial_ptr,
    seq_q, seq_k, batch_size, total_dk: tl.constexpr,
    dim_qk: tl.constexpr, dim_v: tl.constexpr,
    scale: tl.constexpr,
    precision_qk: tl.constexpr, precision_fp32: tl.constexpr,
    precision_do_v: tl.constexpr, input_dtype_code,
    use_atomic: tl.constexpr,
    max_block_m, max_block_n,
    block_m: tl.constexpr, block_n: tl.constexpr,
    block_d: tl.constexpr, block_dv: tl.constexpr,
):
    """Compute each (query, key) tile once for dQ, dK, and dV.

    A CTA owns a query tile, so it can locally accumulate dQ. It contributes
    its dK/dV partials either with FP32 atomics or to a per-query-tile scratch
    buffer for a separate reduction.
    """
    batch = tl.program_id(1)
    pid_m = tl.program_id(0)

    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_n)
    dims = tl.arange(0, block_d)
    dims_v = tl.arange(0, block_dv)

    q = tl.load(
        q_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
        (rows[:, None] < seq_q) & (dims[None, :] < dim_qk),
        other=0,
    )
    do = tl.load(
        do_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
        (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v),
        other=0,
    )
    o = tl.load(
        o_ptr + batch * seq_q * dim_v + rows[:, None] * dim_v + dims_v[None, :],
        (rows[:, None] < seq_q) & (dims_v[None, :] < dim_v),
        other=0,
    ).to(tl.float32)
    lse = tl.load(lse_ptr + batch * seq_q + rows, rows < seq_q, other=-float("inf"))
    safe_lse = tl.where(lse == -float("inf"), 0.0, lse)
    delta = tl.sum(do.to(tl.float32) * o, axis=1)

    dq_acc = tl.zeros((block_m, block_d), tl.float32)
    end_n = tl.minimum((pid_m + 1) * block_m, seq_k)
    for start_n in range(0, end_n, block_n):
        key_cols = start_n + cols
        k = tl.load(
            k_ptr + batch * seq_k * dim_qk + key_cols[:, None] * dim_qk + dims[None, :],
            (key_cols[:, None] < seq_k) & (dims[None, :] < dim_qk),
            other=0,
        )
        v = tl.load(
            v_ptr + batch * seq_k * dim_v + key_cols[:, None] * dim_v + dims_v[None, :],
            (key_cols[:, None] < seq_k) & (dims_v[None, :] < dim_v),
            other=0,
        )
        diagonal_tile = start_n + block_n > pid_m * block_m
        dq_tile, dk_part, dv_part = _attention_backward_tile(
            q, k, v, do, safe_lse, delta, rows, key_cols,
            seq_q, seq_k, scale, precision_qk, precision_fp32,
            precision_do_v, q_ptr.dtype.element_ty, diagonal_tile,
        )
        dq_acc += dq_tile

        dk_offset = batch * seq_k * dim_qk + key_cols[:, None] * dim_qk + dims[None, :]
        dv_offset = total_dk + batch * seq_k * dim_v + key_cols[:, None] * dim_v + dims_v[None, :]
        if use_atomic:
            tl.atomic_add(
                dkv_acc_ptr + dk_offset,
                dk_part,
                mask=(key_cols[:, None] < seq_k) & (dims[None, :] < dim_qk),
                sem="relaxed",
            )
            tl.atomic_add(
                dkv_acc_ptr + dv_offset,
                dv_part,
                mask=(key_cols[:, None] < seq_k) & (dims_v[None, :] < dim_v),
                sem="relaxed",
            )
        else:
            partial_base = dkv_partial_ptr + pid_m * (total_dk + batch_size * seq_k * dim_v)
            tl.store(
                partial_base + dk_offset,
                dk_part,
                mask=(key_cols[:, None] < seq_k) & (dims[None, :] < dim_qk),
            )
            tl.store(
                partial_base + dv_offset,
                dv_part,
                mask=(key_cols[:, None] < seq_k) & (dims_v[None, :] < dim_v),
            )

    tl.store(
        dq_ptr + batch * seq_q * dim_qk + rows[:, None] * dim_qk + dims[None, :],
        dq_acc,
        (rows[:, None] < seq_q) & (dims[None, :] < dim_qk),
    )


@_jit
def _cast_dkv_kernel(
    dkv_acc_ptr, dk_ptr, dv_ptr,
    total_dk, total_dv, block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    total = total_dk + total_dv
    values = tl.load(dkv_acc_ptr + offsets, offsets < total, other=0.0)
    dk_mask = offsets < total_dk
    dv_offsets = tl.maximum(offsets - total_dk, 0)
    tl.store(dk_ptr + offsets, values, dk_mask)
    tl.store(dv_ptr + dv_offsets, values, (offsets >= total_dk) & (offsets < total))


@_jit
def _reduce_dkv_partials_kernel(
    partial_ptr, dk_ptr, dv_ptr,
    total_dk, total_dv, seq_k,
    dim_qk: tl.constexpr, dim_v: tl.constexpr,
    num_query_tiles, block_m: tl.constexpr, block_n: tl.constexpr,
    block_size: tl.constexpr,
):
    """Reduce per-query-tile dK/dV writes without global atomics."""
    num_key_tiles = tl.cdiv(seq_k, block_n)
    batch = tl.program_id(0) // num_key_tiles
    key_tile = tl.program_id(0) % num_key_tiles
    local_offsets = tl.program_id(1) * block_size + tl.arange(0, block_size)

    dk_tile_elements = block_n * dim_qk
    dv_tile_elements = block_n * dim_v
    is_dk = local_offsets < dk_tile_elements
    dk_local_row = local_offsets // dim_qk
    dk_dim = local_offsets % dim_qk
    dv_local = local_offsets - dk_tile_elements
    dv_local_row = dv_local // dim_v
    dv_dim = dv_local % dim_v

    dk_row = key_tile * block_n + dk_local_row
    dv_row = key_tile * block_n + dv_local_row
    dk_offset = batch * seq_k * dim_qk + dk_row * dim_qk + dk_dim
    dv_offset = total_dk + batch * seq_k * dim_v + dv_row * dim_v + dv_dim
    global_offsets = tl.where(is_dk, dk_offset, dv_offset)
    valid = tl.where(
        is_dk,
        (dk_row < seq_k) & (local_offsets < dk_tile_elements),
        (dv_row < seq_k) & (dv_local < dv_tile_elements),
    )

    first_query_tile = (key_tile * block_n) // block_m
    accum = tl.zeros((block_size,), tl.float32)
    for query_tile in range(first_query_tile, num_query_tiles):
        values = tl.load(
            partial_ptr + query_tile * (total_dk + total_dv) + global_offsets,
            valid,
            other=0.0,
        )
        accum += values

    tl.store(dk_ptr + dk_offset, accum, valid & is_dk)
    tl.store(dv_ptr + dv_offset - total_dk, accum, valid & ~is_dk)


class _TritonCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_m, block_n, precision_fp32, dkv_accumulation, warp_strategy):
        batch, num_heads, q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride = (
            _attention_batch_layout(q)
        )
        k_batch, k_heads, k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride = (
            _attention_batch_layout(k)
        )
        v_batch, v_heads, v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride = (
            _attention_batch_layout(v)
        )
        if (batch, num_heads) != (k_batch, k_heads) or (batch, num_heads) != (v_batch, v_heads):
            raise ValueError("Q, K, V must use the same batch/head layout on the Triton path")
        seq_q, dim_qk = q.shape[-2:]
        seq_k = k.shape[-2]
        dim_v = v.shape[-1]
        output = torch.empty((batch, seq_q, dim_v), device=q.device, dtype=q.dtype)
        lse = torch.empty((batch, seq_q), device=q.device, dtype=torch.float32)

        if batch == 0 or seq_q == 0:
            pass
        elif seq_k == 0:
            output.zero_()
            lse.fill_(-torch.inf)
        else:
            scale = 1.0 / math.sqrt(dim_qk)
            precision_qk = "tf32" if q.dtype in (torch.float16, torch.bfloat16) else "tf32x3"
            input_dtype_code = 1 if q.dtype == torch.float16 else 2 if q.dtype == torch.bfloat16 else 0
            block_d = max(16, triton.next_power_of_2(dim_qk))
            block_dv = max(16, triton.next_power_of_2(dim_v))
            kernel = (
                _attention_forward_sliced_q_kernel
                if warp_strategy == "sliced_q"
                else _attention_forward_kernel
            )
            grid = lambda meta: (triton.cdiv(seq_q, meta["block_m"]), batch)
            if warp_strategy == "sliced_q":
                kernel[grid](
                    q, k, v, output, lse,
                    seq_q, seq_k, batch, dim_qk, dim_v, scale,
                    precision_fp32, input_dtype_code,
                    block_m, block_n,
                    block_d=block_d, block_dv=block_dv,
                )
            else:
                kernel[grid](
                    q, k, v, output, lse,
                    seq_q, seq_k, batch, num_heads, dim_qk, dim_v,
                    q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
                    k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
                    v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride,
                    scale,
                    precision_qk, precision_fp32, input_dtype_code,
                    block_m, block_n,
                    block_d=block_d, block_dv=block_dv,
                )

        ctx.save_for_backward(q, k, v, output, lse)
        ctx.max_block_m, ctx.max_block_n = block_m, block_n
        ctx.precision_fp32 = precision_fp32
        ctx.dkv_accumulation = dkv_accumulation
        ctx.warp_strategy = warp_strategy
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Keep AMP's selected operand dtype. FP16/BF16 dots use native Tensor
        # Cores with FP32 accumulators; FP32 dots follow the requested mode.
        with torch.autocast(device_type="cuda", enabled=False):
            return _TritonCausalAttention._backward_impl(ctx, grad_output)

    @staticmethod
    def _backward_impl(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        batch, num_heads, q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride = (
            _attention_batch_layout(q)
        )
        k_batch, k_heads, k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride = (
            _attention_batch_layout(k)
        )
        v_batch, v_heads, v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride = (
            _attention_batch_layout(v)
        )
        seq_q, dim_qk = q.shape[-2:]
        seq_k = k.shape[-2]
        dim_v = v.shape[-1]

        dq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        dk = torch.empty(k.shape, device=k.device, dtype=k.dtype)
        dv = torch.empty(v.shape, device=v.device, dtype=v.dtype)
        if batch == 0 or seq_q == 0 or seq_k == 0:
            dq.zero_()
            dk.zero_()
            dv.zero_()
            return dq, dk, dv, None, None, None, None, None

        max_block_m = ctx.max_block_m
        max_block_n = ctx.max_block_n
        block_d = max(16, triton.next_power_of_2(dim_qk))
        block_dv = max(16, triton.next_power_of_2(dim_v))
        scale = 1.0 / math.sqrt(dim_qk)
        precision_qk = "tf32" if q.dtype in (torch.float16, torch.bfloat16) else "tf32x3"
        precision_fp32 = ctx.precision_fp32
        precision_do_v = (
            "tf32"
            if grad_output.dtype in (torch.float16, torch.bfloat16)
            else "tf32x3"
        )
        input_dtype_code = 1 if q.dtype == torch.float16 else 2 if q.dtype == torch.bfloat16 else 0
        if ctx.dkv_accumulation == "exclusive":
            do_batch_stride, do_seq_stride, do_dim_stride = grad_output.stride()
            delta = torch.empty((batch, seq_q), device=q.device, dtype=torch.float32)
            delta_block_m = 16
            _attention_delta_kernel[
                (triton.cdiv(seq_q, delta_block_m), batch)
            ](
                output, grad_output, delta,
                seq_q, dim_v, batch,
                do_batch_stride, do_seq_stride, do_dim_stride,
                block_m=delta_block_m, block_dv=block_dv,
                num_warps=4,
            )
            dq_acc = torch.zeros(q.numel(), device=q.device, dtype=torch.float32)
            _attention_backward_kv_outer_kernel[
                lambda meta: (triton.cdiv(seq_k, meta["block_n"]), batch)
            ](
                q, k, v, output, lse, grad_output, delta,
                dq_acc, dk, dv,
                seq_q, seq_k, batch, num_heads, dim_qk, dim_v,
                q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
                k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
                v_batch_stride, v_head_stride, v_seq_stride, v_dim_stride,
                scale,
                precision_qk, precision_fp32, precision_do_v, input_dtype_code,
                do_batch_stride, do_seq_stride, do_dim_stride,
                max_block_m, max_block_n,
                block_d=block_d, block_dv=block_dv,
            )
            _cast_fp32_gradient_kernel[(triton.cdiv(q.numel(), 1024),)](
                dq_acc, dq, q.numel(), block_size=1024, num_warps=4,
            )
            return dq, dk, dv, None, None, None, None, None

        grad_output = grad_output.contiguous()

        total_dk = batch * seq_k * dim_qk
        total_dv = batch * seq_k * dim_v
        use_atomic = ctx.dkv_accumulation == "atomic"
        if use_atomic:
            # FP32 scratch keeps cross-query-tile accumulation accurate for
            # BF16 and FP16 inputs. torch.zeros initializes atomic targets.
            dkv_acc = torch.zeros(total_dk + total_dv, device=q.device, dtype=torch.float32)
            dkv_partial = torch.empty(1, device=q.device, dtype=torch.float32)
        else:
            # Reserve enough rows for the smallest autotuned query tile.
            # Each CTA writes only its own row; reduction reads only rows that
            # actually visited each KV tile.
            min_block_m = min(config.kwargs["block_m"] for config in _BACKWARD_CONFIGS)
            max_query_tiles = triton.cdiv(seq_q, min_block_m)
            dkv_acc = torch.empty(1, device=q.device, dtype=torch.float32)
            dkv_partial = torch.empty(
                max_query_tiles * (total_dk + total_dv),
                device=q.device,
                dtype=torch.float32,
            )
        _attention_backward_fused_kernel[
            lambda meta: (triton.cdiv(seq_q, meta["block_m"]), batch)
        ](
            q, k, v, output, lse, grad_output, dq, dkv_acc, dkv_partial,
            seq_q, seq_k, batch, total_dk, dim_qk, dim_v, scale,
            precision_qk, precision_fp32, precision_do_v, input_dtype_code,
            use_atomic,
            max_block_m, max_block_n,
            block_d=block_d, block_dv=block_dv,
        )
        if use_atomic:
            cast_block = 1024
            _cast_dkv_kernel[(triton.cdiv(total_dk + total_dv, cast_block),)](
                dkv_acc, dk, dv, total_dk, total_dv, cast_block,
                num_warps=4,
            )
        else:
            config = _attention_backward_fused_kernel.best_config
            block_m = config.kwargs["block_m"]
            block_n = config.kwargs["block_n"]
            num_query_tiles = triton.cdiv(seq_q, block_m)
            block_size = 1024
            num_key_tiles = triton.cdiv(seq_k, block_n)
            _reduce_dkv_partials_kernel[
                (batch * num_key_tiles, triton.cdiv(block_n * (dim_qk + dim_v), block_size))
            ](
                dkv_partial, dk, dv,
                total_dk, total_dv, seq_k, dim_qk, dim_v,
                num_query_tiles, block_m, block_n, block_size,
                num_warps=4,
            )
        return dq, dk, dv, None, None, None, None, None


def block_causal_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_block_size: int = 64,
    kv_block_size: int = 64,
    *,
    attn_mask: torch.Tensor | None = None,
    fp32_dot_precision: str = "tf32x3",
    dkv_accumulation: str = "exclusive",
    warp_strategy: str = "compiler",
):
    """Compute upper-left causal attention for Q (..., L, D), K (..., S, D), V (..., S, Dv).

    On CUDA with Triton and float16, bfloat16, or float32 inputs, one Triton
    kernel performs the online-softmax QK and PV calculation. Backward
    recomputes probability tiles from Q, K, V, O and LSE, without retaining a
    full attention matrix. The default backward precomputes row-wise delta,
    then assigns one CTA to each KV tile. It accumulates dK/dV in registers
    and atomically accumulates dQ into an FP32 workspace.
    The Triton path supports rank-2, rank-3, and rank-4 inputs with matching
    leading dimensions, including non-contiguous Q/K/V strides. Other cases,
    including broadcasted leading dimensions and explicit ``attn_mask``
    inputs, use the PyTorch reference path.

    ``fp32_dot_precision`` selects ``tf32x3`` (default, higher precision),
    ``tf32`` (single TF32 dot), or ``bf16`` (round intermediates to BF16) for
    dots involving FP32 intermediates. ``dkv_accumulation="exclusive"`` uses
    the KV-owned traversal with register-local dK/dV accumulation. ``atomic``
    and ``partial`` keep the fused Q-owned path for comparisons.
    ``warp_strategy="sliced_q"`` uses an experimental Gluon forward
    kernel that tunes the MMA warp partition across query and key/value axes.
    Its ``warps_m=4`` candidate maps the four warps to separate query-row
    slices, matching FA2 Sliced-Q; ``"compiler"`` keeps regular Triton layout
    selection. These controls support CUDA performance/accuracy tests.
    """
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("attention block sizes must be positive")
    if fp32_dot_precision not in ("tf32", "tf32x3", "bf16"):
        raise ValueError("fp32_dot_precision must be 'tf32x3', 'tf32', or 'bf16'")
    if dkv_accumulation not in ("exclusive", "atomic", "partial"):
        raise ValueError("dkv_accumulation must be 'exclusive', 'atomic', or 'partial'")
    if warp_strategy not in ("compiler", "sliced_q"):
        raise ValueError("warp_strategy must be 'compiler' or 'sliced_q'")
    if warp_strategy == "sliced_q":
        if _attention_forward_sliced_q_kernel is None:
            raise RuntimeError("warp_strategy='sliced_q' requires Triton Gluon")
        if q_block_size < 64:
            raise ValueError("warp_strategy='sliced_q' requires q_block_size >= 64")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("warp_strategy='sliced_q' currently supports float16 and bfloat16")

    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError("Q, K, V must each have at least two dimensions")
    if q.shape[-1] == 0 or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K must have the same positive feature dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("K and V must have the same sequence length")
    if not q.is_floating_point() or q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q, K, V must share a floating-point dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, V must be on the same device")
    try:
        batch_shape = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    except RuntimeError as error:
        raise ValueError("Q, K, V leading dimensions must be broadcastable") from error

    # Keep the reference implementation's full mask behavior. The fused path
    # intentionally saves exactly Q, K, V, O, and LSE.
    if (
        attn_mask is not None
        or triton is None
        or q.device.type != "cuda"
        or q.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or q.shape[:-2] != batch_shape
        or k.shape[:-2] != batch_shape
        or v.shape[:-2] != batch_shape
        or q.ndim not in (2, 3, 4)
        or k.ndim != q.ndim
        or v.ndim != q.ndim
    ):
        return _torch_block_attention(
            q, k, v, q_block_size, kv_block_size, attn_mask=attn_mask
        )

    seq_q, dim_qk = q.shape[-2:]
    seq_k, dim_v = k.shape[-2], v.shape[-1]
    batch = math.prod(batch_shape) if batch_shape else 1
    inputs_contiguous = q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    # The older Q-owned comparison kernels and the Gluon warp-layout kernel
    # currently consume flattened contiguous operands. The default KV-owned
    # path accepts arbitrary strides and reads each logical head in-kernel.
    if (
        (dkv_accumulation != "exclusive" or warp_strategy == "sliced_q")
        and not inputs_contiguous
    ):
        return _torch_block_attention(
            q, k, v, q_block_size, kv_block_size, attn_mask=attn_mask
        )

    # Triton ranges and dot tiles use powers of two (and dot dimensions of at
    # least 16). Round user-provided tile sizes up while bounding row/column
    # tiles to keep register use manageable.
    block_m = max(16, min(128, triton.next_power_of_2(q_block_size)))
    block_n = max(16, min(128, triton.next_power_of_2(kv_block_size)))
    # Flatten contiguous inputs as views. For transposed/non-contiguous inputs,
    # preserve the original tensor and pass its strides to the kernel.
    if inputs_contiguous:
        q_kernel = q.view(batch, seq_q, dim_qk)
        k_kernel = k.view(batch, seq_k, dim_qk)
        v_kernel = v.view(batch, seq_k, dim_v)
    else:
        q_kernel, k_kernel, v_kernel = q, k, v
    # Keep AMP-selected input dtypes intact for native FP16/BF16 Tensor Core
    # dots; FP32 accumulation and output casts happen inside the kernels.
    with torch.autocast(device_type="cuda", enabled=False):
        output = _TritonCausalAttention.apply(
            q_kernel, k_kernel, v_kernel, block_m, block_n,
            fp32_dot_precision, dkv_accumulation, warp_strategy,
        )
    return output.view(batch_shape + (seq_q, dim_v))


# A shorter alias is convenient when this module is imported as the attention
# implementation directly.
block_attention_triton = block_causal_attention_triton
block_causal_attention = block_causal_attention_triton

__all__ = [
    "block_causal_attention_triton",
    "block_attention_triton",
    "block_causal_attention",
]
