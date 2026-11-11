"""Causal tiled attention with online softmax and a manual first-order backward.

Scores/probabilities are materialized only for a Q-row/KV-column tile. The
backward recomputes tiles from saved Q/K/V, output and per-row logsumexp.
This is a PyTorch implementation, not a fused CUDA kernel.
"""
import math

import torch
from torch.autograd.function import once_differentiable


def _scores(q, k, row_start, col_start, scale, query_positions,
            key_positions, attn_mask=None):
    scores = (q @ k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        tile = attn_mask[..., row_start:row_start + q.shape[-2],
                         col_start:col_start + k.shape[-2]]
        if tile.dtype == torch.bool:
            scores = scores.masked_fill(~tile, -torch.inf)
        else:
            scores = scores + tile.to(scores.dtype)
    # Most KV tiles are wholly before this query tile and need no causal mask.
    # Reuse the position vectors created once per forward/backward call instead
    # of launching two arange kernels for every score tile.
    last_col = col_start + k.shape[-2] - 1
    if last_col > row_start:
        rows = query_positions[row_start:row_start + q.shape[-2]]
        cols = key_positions[col_start:col_start + k.shape[-2]]
        scores = scores.masked_fill(cols[None, :] > rows[:, None], -torch.inf)
    return scores


class _BlockAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_block_size, kv_block_size, attn_mask):
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            qf, kf, vf = q.to(dtype), k.to(dtype), v.to(dtype)
            n, d = q.shape[-2:]
            scale = 1 / math.sqrt(d)
            output = torch.zeros(q.shape[:-1] + (v.shape[-1],), device=q.device, dtype=dtype)
            logsumexp = torch.empty(q.shape[:-1], device=q.device, dtype=dtype)
            query_positions = torch.arange(q.shape[-2], device=q.device)
            key_positions = torch.arange(k.shape[-2], device=k.device)
            for i in range(0, n, q_block_size):
                qi = qf[..., i:i + q_block_size, :]
                shape = qi.shape[:-1] + (1,)
                maximum = torch.full(shape, -torch.inf, device=q.device, dtype=dtype)
                normalizer = torch.zeros(shape, device=q.device, dtype=dtype)
                accumulator = torch.zeros(qi.shape[:-1] + (v.shape[-1],), device=q.device, dtype=dtype)
                kv_stop = min(i + qi.shape[-2], k.shape[-2])
                for j in range(0, kv_stop, kv_block_size):
                    kj, vj = kf[..., j:j + kv_block_size, :], vf[..., j:j + kv_block_size, :]
                    scores = _scores(qi, kj, i, j, scale, query_positions,
                                     key_positions, attn_mask)
                    new_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
                    shift = new_max.masked_fill(torch.isneginf(new_max), 0)
                    alpha = (maximum - shift).exp()
                    probabilities = (scores - shift).exp()
                    accumulator = accumulator * alpha + probabilities @ vj
                    normalizer = normalizer * alpha + probabilities.sum(-1, keepdim=True)
                    maximum = new_max
                output[..., i:i + q_block_size, :] = accumulator / normalizer.clamp_min(torch.finfo(dtype).tiny)
                logsumexp[..., i:i + q_block_size] = (maximum + normalizer.log()).squeeze(-1)
            ctx.save_for_backward(q, k, v, output, logsumexp, attn_mask)
            ctx.q_block_size, ctx.kv_block_size = q_block_size, kv_block_size
            return output.to(q.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        q, k, v, output, logsumexp, attn_mask = ctx.saved_tensors
        with torch.autocast(device_type=q.device.type, enabled=False):
            dtype = output.dtype
            qf, kf, vf, go = q.to(dtype), k.to(dtype), v.to(dtype), grad_output.to(dtype)
            dq, dk, dv = torch.zeros_like(qf), torch.zeros_like(kf), torch.zeros_like(vf)
            dm = (torch.zeros_like(attn_mask, dtype=dtype)
                  if attn_mask is not None and ctx.needs_input_grad[5] else None)
            n, d = q.shape[-2:]
            scale = 1 / math.sqrt(d)
            query_positions = torch.arange(q.shape[-2], device=q.device)
            key_positions = torch.arange(k.shape[-2], device=k.device)
            # Sum_j dP_ij P_ij = dot(dO_i, O_i).
            delta = (go * output).sum(-1, keepdim=True)
            for i in range(0, n, ctx.q_block_size):
                qi = qf[..., i:i + ctx.q_block_size, :]
                doi = go[..., i:i + ctx.q_block_size, :]
                lse = logsumexp[..., i:i + ctx.q_block_size, None]
                lse = lse.masked_fill(torch.isneginf(lse), 0)
                di = delta[..., i:i + ctx.q_block_size, :]
                kv_stop = min(i + qi.shape[-2], k.shape[-2])
                for j in range(0, kv_stop, ctx.kv_block_size):
                    kj = kf[..., j:j + ctx.kv_block_size, :]
                    vj = vf[..., j:j + ctx.kv_block_size, :]
                    p = (_scores(qi, kj, i, j, scale, query_positions,
                                 key_positions, attn_mask) - lse).exp()
                    dp = doi @ vj.transpose(-2, -1)
                    ds = p * (dp - di)
                    if dm is not None:
                        dm[..., i:i + ctx.q_block_size, j:j + ctx.kv_block_size] = ds
                    dq[..., i:i + ctx.q_block_size, :].add_((ds @ kj) * scale)
                    dk[..., j:j + ctx.kv_block_size, :].add_((ds.transpose(-2, -1) @ qi) * scale)
                    dv[..., j:j + ctx.kv_block_size, :].add_(p.transpose(-2, -1) @ doi)
            return (dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None,
                    dm.to(attn_mask.dtype) if dm is not None else None)


def block_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           q_block_size: int = 64, kv_block_size: int = 64, *,
                           attn_mask: torch.Tensor | None = None):
    """Causal attention for Q (..., L, D), K (..., S, D), V (..., S, Dv).

    Leading dimensions broadcast; output has shape (..., L, Dv). Causality is
    upper-left aligned: query i attends to keys j <= i, also when L != S.
    Empty key/value sequences produce zero outputs.
    attn_mask must broadcast to (..., L, S). Boolean True allows attention;
    floating masks are added to scores (use -inf to block). The causal mask
    always applies. Fully masked rows return zero with zero gradients.
    Floating masks support first-order gradients.

    Low-precision accumulation uses float32; float64 is preserved for gradient
    checks. Supports arbitrary non-contiguous strides and partial final tiles.
    Only first-order derivatives are supported by the manual backward.
    """
    if q_block_size <= 0 or kv_block_size <= 0:
        raise ValueError("attention block sizes must be positive")
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
    q = q.expand(batch_shape + q.shape[-2:])
    k = k.expand(batch_shape + k.shape[-2:])
    v = v.expand(batch_shape + v.shape[-2:])
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool and not attn_mask.is_floating_point():
            raise TypeError("attn_mask must be boolean or floating-point")
        if attn_mask.device != q.device:
            raise ValueError("attn_mask must be on the same device as Q")
        try:
            attn_mask = attn_mask.expand(batch_shape + (q.shape[-2], k.shape[-2]))
        except RuntimeError as error:
            raise ValueError("attn_mask must broadcast to (..., query_length, key_length)") from error
    return _BlockAttention.apply(q, k, v, q_block_size, kv_block_size, attn_mask)
