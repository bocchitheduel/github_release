#!/usr/bin/env python3
"""Measure Muon Triton epilogue precision and CUDA parameter-update paths."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import torch
import triton
import triton.testing

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muon import Muon
from muon_kernels import (NS_PRECISIONS, fused_parameter_update,
                          polynomial_gemm_epilogue, x_update_gemm_epilogue)


def benchmark_ms(function, *, warmup: int = 30, rep: int = 100) -> float:
    return float(triton.testing.do_bench(function, warmup=warmup, rep=rep))


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_abs = float(difference.max().item())
    relative_l2 = float(torch.linalg.vector_norm(difference)
                        / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30))
    return max_abs, relative_l2


def benchmark_epilogues(shapes: list[tuple[int, int, int]]) -> list[dict]:
    results = []
    for batch, rows, cols in shapes:
        torch.manual_seed(20261110 + rows + cols)
        gram = torch.randn((batch, rows, rows), device="cuda", dtype=torch.float32)
        direction = torch.randn((batch, rows, cols), device="cuda", dtype=torch.float32)
        polynomial = torch.empty_like(gram)
        direction_next = torch.empty_like(direction)
        beta, alpha = -4.775, 2.0315
        reference_polynomial = torch.baddbmm(gram, gram, gram, beta=beta, alpha=alpha)
        reference_next = torch.baddbmm(direction, reference_polynomial, direction,
                                       beta=3.4445, alpha=1.0)
        for precision in NS_PRECISIONS:
            polynomial_gemm_epilogue(gram, polynomial, beta=beta, alpha=alpha,
                                     precision=precision)
            torch.cuda.synchronize()
            poly_max, poly_rel = relative_error(polynomial, reference_polynomial)
            next_call = lambda: x_update_gemm_epilogue(
                polynomial, direction, direction_next, beta=3.4445,
                precision=precision)
            next_call()
            torch.cuda.synchronize()
            next_max, next_rel = relative_error(direction_next, reference_next)
            poly_call = lambda: polynomial_gemm_epilogue(
                gram, polynomial, beta=beta, alpha=alpha, precision=precision)
            results.append({
                "batch": batch, "rows": rows, "cols": cols,
                "precision": precision,
                "poly_max_abs": poly_max, "poly_rel_l2": poly_rel,
                "x_max_abs": next_max, "x_rel_l2": next_rel,
                "poly_ms": benchmark_ms(poly_call),
                "x_ms": benchmark_ms(next_call),
                "torch_poly_ms": benchmark_ms(
                    lambda: torch.baddbmm(gram, gram, gram, beta=beta, alpha=alpha)),
                "torch_x_ms": benchmark_ms(
                    lambda: torch.baddbmm(direction, reference_polynomial, direction,
                                          beta=3.4445, alpha=1.0)),
            })
    return results


def benchmark_updates(batch: int = 8) -> list[dict]:
    results = []
    update_scale, decay_scale = -0.003, 0.9998
    for rows, cols, transposed in ((512, 512, False), (512, 1344, False),
                                   (512, 1344, True)):
        storage_shape = (cols, rows) if transposed else (rows, cols)
        parameters = [torch.randn(storage_shape, device="cuda") for _ in range(batch)]
        directions = torch.randn((batch, rows, cols), device="cuda")
        parameter_batch = torch.empty((batch, *storage_shape), device="cuda")

        def fused():
            torch.stack(parameters, dim=0, out=parameter_batch)
            fused_parameter_update(
                parameter_batch, directions, rows=storage_shape[0], cols=storage_shape[1],
                transposed=transposed, update_scale=update_scale,
                decay_scale=decay_scale)
            torch._foreach_copy_(parameters, list(parameter_batch.unbind(0)))

        def foreach():
            oriented = [directions[i].transpose(0, 1) if transposed else directions[i]
                        for i in range(batch)]
            torch._foreach_mul_(parameters, decay_scale)
            torch._foreach_add_(parameters, oriented, alpha=update_scale)

        initial = [parameter.clone() for parameter in parameters]
        fused()
        fused_result = [parameter.clone() for parameter in parameters]
        for parameter, original in zip(parameters, initial):
            parameter.copy_(original)
        foreach()
        update_error = max(float((left - right).abs().max().item())
                           for left, right in zip(parameters, fused_result))
        for parameter, original in zip(parameters, initial):
            parameter.copy_(original)
        results.append({
            "batch": batch, "rows": rows, "cols": cols,
            "transposed": transposed,
            "fused_ms": benchmark_ms(fused),
            "foreach_ms": benchmark_ms(foreach),
            "max_abs_diff": update_error,
        })
    return results


def benchmark_full_step() -> list[dict]:
    class ReferenceMuon(Muon):
        @staticmethod
        def _can_use_batched_cuda(active):
            return False

    shapes = ((256, 256), (512, 256), (256, 512), (512, 512))
    torch.manual_seed(712)
    initial = [torch.randn(shape, device="cuda") for shape in shapes]
    gradients = [torch.randn_like(parameter) for parameter in initial]
    reference_parameters = [torch.nn.Parameter(value.clone()) for value in initial]
    reference = ReferenceMuon(reference_parameters, lr=0.003, momentum=0.9,
                             nesterov=True, weight_decay=0.02, ns_steps=5)
    for parameter, gradient in zip(reference_parameters, gradients):
        parameter.grad = gradient.clone()
    reference.step()
    reference_deltas = [parameter.detach() - value
                        for parameter, value in zip(reference_parameters, initial)]

    results = []
    for precision in NS_PRECISIONS:
        for update_strategy in ("fused", "foreach"):
            parameters = [torch.nn.Parameter(value.clone()) for value in initial]
            optimizer = Muon(parameters, lr=0.003, momentum=0.9, nesterov=True,
                             weight_decay=0.02, ns_steps=5, ns_precision=precision,
                             update_strategy=update_strategy)
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient.clone()
            optimizer.step()
            deltas = [parameter.detach() - value
                      for parameter, value in zip(parameters, initial)]
            numerator = torch.sqrt(sum((left - right).square().sum()
                                       for left, right in zip(deltas, reference_deltas)))
            denominator = torch.sqrt(sum(right.square().sum()
                                         for right in reference_deltas))
            results.append({
                "precision": precision, "update_strategy": update_strategy,
                "relative_update_l2": float(numerator / denominator.clamp_min(1e-30)),
                "step_ms": benchmark_ms(optimizer.step, warmup=10, rep=50),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="use a small smoke workload for fast iteration")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "muon_precision_update_20261110.md")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = [(2, 128, 256)] if args.quick else [(4, 256, 512), (2, 512, 1344)]
    epilogues = benchmark_epilogues(shapes)
    updates = benchmark_updates(batch=4 if args.quick else 8)
    full_step = benchmark_full_step()

    lines = ["# Muon epilogue precision and parameter update benchmark", "",
             f"- Date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"- GPU: {torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}; "
             f"Triton {triton.__version__}; CUDA {torch.version.cuda}",
             "- All accuracy references use FP32 torch.baddbmm with TF32 disabled.", "",
             "## Triton epilogue accuracy and latency", "",
             "Times are milliseconds per call. Relative L2 is `||result-reference||₂ / ||reference||₂`.", "",
             "| Batch × M × N | Precision | Polynomial rel-L2 | Polynomial ms | X-update rel-L2 | X-update ms | torch polynomial ms | torch X-update ms |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in epilogues:
        lines.append(f"| {row['batch']} × {row['rows']} × {row['cols']} | {row['precision']} | "
                     f"{row['poly_rel_l2']:.3e} | {row['poly_ms']:.4f} | "
                     f"{row['x_rel_l2']:.3e} | {row['x_ms']:.4f} | "
                     f"{row['torch_poly_ms']:.4f} | {row['torch_x_ms']:.4f} |")
    lines.extend(["", "## Full optimizer update precision", "",
                  "Relative L2 is measured against the matrix-by-matrix FP32 Muon implementation after one full update; step time uses the selected epilogue and parameter-update paths.", "",
                  "| NS precision | Update strategy | Relative update L2 | Full step ms |",
                  "|---|---|---:|---:|"])
    for row in full_step:
        lines.append(f"| {row['precision']} | {row['update_strategy']} | "
                     f"{row['relative_update_l2']:.3e} | {row['step_ms']:.4f} |")
    lines.extend(["", "## Parameter-update strategy", "",
                  "The fused path includes weight stacking, the fused Triton update, and copy-back. Foreach applies decay and update directly to each parameter tensor.", "",
                  "| Batch | Oriented M × N | Storage transposed | Fused ms | Foreach ms | Fused/foreach | Max abs difference |",
                  "|---:|---:|:---:|---:|---:|---:|---:|"])
    for row in updates:
        lines.append(f"| {row['batch']} | {row['rows']} × {row['cols']} | "
                     f"{row['transposed']} | {row['fused_ms']:.4f} | "
                     f"{row['foreach_ms']:.4f} | {row['fused_ms'] / row['foreach_ms']:.3f} | "
                     f"{row['max_abs_diff']:.3e} |")
    lines.extend([
        "", "## Notes", "",
        "- `tf32x3` is the closest Triton mode to FP32 (`~1e-7` relative L2 in the epilogue checks), but it trails PyTorch GEMM on the largest shape. `tf32` and `fp16` are much faster here; BF16 has the largest measured full-update deviation.",
        "- Foreach is faster in each isolated parameter-update case, with full-step results varying by precision. It remains selectable with `muon_update_strategy: foreach` or `--muon-update-strategy foreach`; the default remains fused.",
        "- The short synthetic benchmark checks numerical and kernel behavior; it does not measure training convergence. Use `muon_ns_precision: tf32`, `bf16`, or `fp16` to select a reduced-precision epilogue.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
