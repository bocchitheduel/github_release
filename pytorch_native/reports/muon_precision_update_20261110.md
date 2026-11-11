# Muon epilogue precision and parameter update benchmark

- Date: 2026-11-10T18:05:49+08:00
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU; PyTorch 2.14.0+cu126; Triton 3.8.0; CUDA 12.6
- All accuracy references use FP32 torch.baddbmm with TF32 disabled.

## Triton epilogue accuracy and latency

Times are milliseconds per call. Relative L2 is `||result-reference||₂ / ||reference||₂`.

| Batch × M × N | Precision | Polynomial rel-L2 | Polynomial ms | X-update rel-L2 | X-update ms | torch polynomial ms | torch X-update ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 × 256 × 512 | tf32x3 | 3.318e-07 | 0.0573 | 4.954e-07 | 0.1074 | 0.0550 | 0.0605 |
| 4 × 256 × 512 | tf32 | 7.629e-04 | 0.0168 | 1.467e-03 | 0.0270 | 0.0511 | 0.0578 |
| 4 × 256 × 512 | bf16 | 2.331e-03 | 0.0156 | 3.315e-03 | 0.0266 | 0.0484 | 0.0560 |
| 4 × 256 × 512 | fp16 | 2.915e-04 | 0.0149 | 4.137e-04 | 0.0269 | 0.0460 | 0.0557 |
| 2 × 512 × 1344 | tf32x3 | 4.404e-07 | 0.1562 | 6.442e-07 | 0.4117 | 0.0890 | 0.2096 |
| 2 × 512 × 1344 | tf32 | 7.670e-04 | 0.0472 | 1.471e-03 | 0.1031 | 0.0927 | 0.1794 |
| 2 × 512 × 1344 | bf16 | 2.339e-03 | 0.0470 | 3.320e-03 | 0.0998 | 0.0841 | 0.1790 |
| 2 × 512 × 1344 | fp16 | 2.922e-04 | 0.0463 | 4.145e-04 | 0.1013 | 0.0833 | 0.1777 |

## Full optimizer update precision

Relative L2 is measured against the matrix-by-matrix FP32 Muon implementation after one full update; step time uses the selected epilogue and parameter-update paths.

| NS precision | Update strategy | Relative update L2 | Full step ms |
|---|---|---:|---:|
| tf32x3 | fused | 2.942e-04 | 3.1721 |
| tf32x3 | foreach | 2.538e-05 | 3.3308 |
| tf32 | fused | 2.210e-03 | 1.9918 |
| tf32 | foreach | 2.203e-03 | 2.0119 |
| bf16 | fused | 1.562e-02 | 2.5738 |
| bf16 | foreach | 1.562e-02 | 2.1224 |
| fp16 | fused | 1.953e-03 | 2.1902 |
| fp16 | foreach | 1.945e-03 | 2.5894 |

## Parameter-update strategy

The fused path includes weight stacking, the fused Triton update, and copy-back. Foreach applies decay and update directly to each parameter tensor.

| Batch | Oriented M × N | Storage transposed | Fused ms | Foreach ms | Fused/foreach | Max abs difference |
|---:|---:|:---:|---:|---:|---:|---:|
| 8 | 512 × 512 | False | 0.1512 | 0.1106 | 1.368 | 4.768e-07 |
| 8 | 512 × 1344 | False | 0.3779 | 0.2797 | 1.351 | 4.768e-07 |
| 8 | 512 × 1344 | True | 0.4169 | 0.3649 | 1.143 | 4.768e-07 |

## Notes

- `tf32x3` is the closest Triton mode to FP32 (`~1e-7` relative L2 in the epilogue checks), but it trails PyTorch GEMM on the largest shape. `tf32` and `fp16` are much faster here; BF16 has the largest measured full-update deviation.
- Foreach is faster in each isolated parameter-update case, with full-step results varying by precision. It remains selectable with `muon_update_strategy: foreach` or `--muon-update-strategy foreach`; the default remains fused.
- The short synthetic benchmark checks numerical and kernel behavior; it does not measure training convergence. Use `muon_ns_precision: tf32`, `bf16`, or `fp16` to select a reduced-precision epilogue.
