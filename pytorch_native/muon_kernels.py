"""CUDA kernels used by the batched Muon optimizer path."""
from __future__ import annotations

import triton
import triton.language as tl


NS_PRECISIONS = ("tf32x3", "tf32", "bf16", "fp16")


@triton.jit
def _finite_partials_kernel(
    gradient_ptr,
    finite_ptr,
    total_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(gradient_ptr + offsets, offsets < total_elements, other=0)
    is_finite = (values == values) & (tl.abs(values) != float("inf"))
    tl.store(finite_ptr + block, tl.min(is_finite.to(tl.int32), 0))


@triton.jit
def _momentum_norm_partials_kernel(
    gradient_ptr,
    momentum_ptr,
    direction_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    ELEMENTS_PER_MATRIX: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    MOMENTUM: tl.constexpr,
    NESTEROV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    matrix = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS_PER_MATRIX
    linear_offsets = matrix * ELEMENTS_PER_MATRIX + offsets
    gradient = tl.load(gradient_ptr + linear_offsets, mask, other=0).to(tl.float32)
    old_momentum = tl.load(momentum_ptr + linear_offsets, mask, other=0)
    new_momentum = MOMENTUM * old_momentum + gradient
    tl.store(momentum_ptr + linear_offsets, new_momentum, mask)

    if NESTEROV:
        direction = gradient + MOMENTUM * new_momentum
    else:
        direction = new_momentum
    tl.store(direction_ptr + linear_offsets, direction, mask)

    magnitude = tl.max(tl.abs(tl.where(mask, direction, 0.0)), 0)
    safe_magnitude = tl.maximum(magnitude, 1.1754943508222875e-38)
    scaled = tl.where(mask, direction / safe_magnitude, 0.0)
    sum_squares = tl.sum(scaled * scaled, 0)
    partial_offset = matrix * NUM_BLOCKS + block
    tl.store(partial_max_ptr + partial_offset, magnitude)
    tl.store(partial_sum_ptr + partial_offset, sum_squares)


@triton.jit
def _norm_scales_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    inverse_scale_ptr,
    NUM_BLOCKS: tl.constexpr,
    EPS: tl.constexpr,
    PARTIALS: tl.constexpr,
):
    matrix = tl.program_id(0)
    offsets = tl.arange(0, PARTIALS)
    mask = offsets < NUM_BLOCKS
    partial_offset = matrix * NUM_BLOCKS + offsets
    maxima = tl.load(partial_max_ptr + partial_offset, mask, other=0.0)
    sums = tl.load(partial_sum_ptr + partial_offset, mask, other=0.0)
    magnitude = tl.max(maxima, 0)
    safe_magnitude = tl.maximum(magnitude, 1.1754943508222875e-38)
    ratios = tl.where(mask, maxima / safe_magnitude, 0.0)
    squared_norm = tl.sqrt(tl.sum(sums * ratios * ratios, 0))
    denominator = squared_norm + EPS / safe_magnitude
    inverse_scale = (1.0 / safe_magnitude) / denominator
    tl.store(inverse_scale_ptr + matrix, inverse_scale)


@triton.jit
def _normalize_direction_kernel(
    direction_ptr,
    inverse_scale_ptr,
    ELEMENTS_PER_MATRIX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    matrix = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS_PER_MATRIX
    linear_offsets = matrix * ELEMENTS_PER_MATRIX + offsets
    values = tl.load(direction_ptr + linear_offsets, mask, other=0.0)
    inverse_scale = tl.load(inverse_scale_ptr + matrix)
    tl.store(direction_ptr + linear_offsets, values * inverse_scale, mask)


@triton.jit
def _fused_parameter_update_kernel(
    parameter_ptr,
    direction_ptr,
    ELEMENTS_PER_MATRIX: tl.constexpr,
    PARAM_ROWS: tl.constexpr,
    PARAM_COLS: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    UPDATE_SCALE: tl.constexpr,
    DECAY_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    matrix = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEMENTS_PER_MATRIX
    if TRANSPOSED:
        source_offsets = (offsets % PARAM_COLS) * PARAM_ROWS + offsets // PARAM_COLS
    else:
        source_offsets = offsets
    parameter_offsets = matrix * ELEMENTS_PER_MATRIX + offsets
    direction_offsets = matrix * ELEMENTS_PER_MATRIX + source_offsets
    parameter = tl.load(parameter_ptr + parameter_offsets, mask, other=0.0).to(tl.float32)
    direction = tl.load(direction_ptr + direction_offsets, mask, other=0.0)
    updated = parameter * DECAY_SCALE + direction * UPDATE_SCALE
    tl.store(parameter_ptr + parameter_offsets, updated, mask)


@triton.jit
def _polynomial_gemm_epilogue_kernel(
    gram_ptr,
    polynomial_ptr,
    M: tl.constexpr,
    BETA: tl.constexpr,
    ALPHA: tl.constexpr,
    PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch = tl.program_id(2)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(tl.cdiv(M, BLOCK_K)):
        ks = k_block * BLOCK_K + reduction
        left = tl.load(
            gram_ptr + batch * M * M + rows[:, None] * M + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < M), other=0.0)
        right = tl.load(
            gram_ptr + batch * M * M + ks[:, None] * M + cols[None, :],
            (ks[:, None] < M) & (cols[None, :] < M), other=0.0)
        if PRECISION == "bf16":
            accumulator = tl.dot(left.to(tl.bfloat16), right.to(tl.bfloat16), accumulator)
        elif PRECISION == "fp16":
            accumulator = tl.dot(left.to(tl.float16), right.to(tl.float16), accumulator)
        elif PRECISION == "tf32":
            accumulator = tl.dot(left, right, accumulator, input_precision="tf32")
        else:
            accumulator = tl.dot(left, right, accumulator, input_precision="tf32x3")

    output_offsets = batch * M * M + rows[:, None] * M + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < M)
    gram = tl.load(gram_ptr + output_offsets, mask, other=0.0)
    result = BETA * gram + ALPHA * accumulator
    tl.store(polynomial_ptr + output_offsets, result, mask)


@triton.jit
def _x_update_gemm_epilogue_kernel(
    polynomial_ptr,
    direction_ptr,
    direction_next_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BETA: tl.constexpr,
    PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    batch = tl.program_id(2)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(tl.cdiv(M, BLOCK_K)):
        ks = k_block * BLOCK_K + reduction
        left = tl.load(
            polynomial_ptr + batch * M * M + rows[:, None] * M + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < M), other=0.0)
        right = tl.load(
            direction_ptr + batch * M * N + ks[:, None] * N + cols[None, :],
            (ks[:, None] < M) & (cols[None, :] < N), other=0.0)
        if PRECISION == "bf16":
            accumulator = tl.dot(left.to(tl.bfloat16), right.to(tl.bfloat16), accumulator)
        elif PRECISION == "fp16":
            accumulator = tl.dot(left.to(tl.float16), right.to(tl.float16), accumulator)
        elif PRECISION == "tf32":
            accumulator = tl.dot(left, right, accumulator, input_precision="tf32")
        else:
            accumulator = tl.dot(left, right, accumulator, input_precision="tf32x3")

    output_offsets = batch * M * N + rows[:, None] * N + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    old_direction = tl.load(direction_ptr + output_offsets, mask, other=0.0)
    result = accumulator + BETA * old_direction
    tl.store(direction_next_ptr + output_offsets, result, mask)


def check_finite(gradient, flags, *, block_size: int = 4096) -> None:
    blocks = triton.cdiv(gradient.numel(), block_size)
    _finite_partials_kernel[(blocks,)](
        gradient, flags, gradient.numel(), block_size, num_warps=4)


def momentum_and_partial_norm(
    gradient,
    momentum,
    direction,
    partial_max,
    partial_sum,
    *,
    momentum_coefficient: float,
    nesterov: bool,
    block_size: int = 1024,
) -> None:
    batch, rows, cols = direction.shape
    elements = rows * cols
    blocks = triton.cdiv(elements, block_size)
    _momentum_norm_partials_kernel[(blocks, batch)](
        gradient, momentum, direction, partial_max, partial_sum,
        elements, blocks, momentum_coefficient, nesterov, block_size,
        num_warps=4)


def normalize_direction(direction, partial_max, partial_sum, inverse_scale, *,
                        eps: float, block_size: int = 1024) -> None:
    batch, rows, cols = direction.shape
    elements = rows * cols
    blocks = triton.cdiv(elements, block_size)
    partials = triton.next_power_of_2(blocks)
    _norm_scales_kernel[(batch,)](
        partial_max, partial_sum, inverse_scale, blocks, eps, partials,
        num_warps=4)
    _normalize_direction_kernel[(blocks, batch)](
        direction, inverse_scale, elements, block_size, num_warps=4)


def fused_parameter_update(parameter_batch, direction, *, rows: int, cols: int,
                           transposed: bool, update_scale: float,
                           decay_scale: float, block_size: int = 1024) -> None:
    batch = parameter_batch.shape[0]
    elements = rows * cols
    blocks = triton.cdiv(elements, block_size)
    _fused_parameter_update_kernel[(blocks, batch)](
        parameter_batch, direction, elements, rows, cols, transposed,
        update_scale, decay_scale, block_size, num_warps=4)


def polynomial_gemm_epilogue(gram, polynomial, *, beta: float, alpha: float,
                             precision: str = "tf32x3",
                             block_m: int = 32, block_n: int = 64,
                             block_k: int = 32) -> None:
    if precision not in NS_PRECISIONS:
        raise ValueError(f"precision must be one of {NS_PRECISIONS}")
    batch, rows, _ = gram.shape
    _polynomial_gemm_epilogue_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(rows, block_n), batch)
    ](gram, polynomial, rows, beta, alpha, precision, block_m, block_n, block_k,
      num_warps=4, num_stages=3)


def x_update_gemm_epilogue(polynomial, direction, direction_next, *, beta: float,
                           precision: str = "tf32x3",
                           block_m: int = 32, block_n: int = 64,
                           block_k: int = 32) -> None:
    if precision not in NS_PRECISIONS:
        raise ValueError(f"precision must be one of {NS_PRECISIONS}")
    batch, rows, cols = direction.shape
    _x_update_gemm_epilogue_kernel[
        (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n), batch)
    ](polynomial, direction, direction_next, rows, cols, beta, precision,
      block_m, block_n, block_k, num_warps=4, num_stages=3)
