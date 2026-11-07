# Version comparison

| Area | `original/` | `pytorch_native/` |
| --- | --- | --- |
| Causal attention | Hand-written tiled attention is the model default; SDPA is available as a selectable backend | PyTorch SDPA is the default; `--attention-backend block` selects the hand-written implementation |
| Optimizer | Project's AdamW implementation | `torch.optim.AdamW`, with `fused=True` on CUDA and the regular PyTorch implementation elsewhere |
| Gradient clipping | Project's implementation | `torch.nn.utils.clip_grad_norm_` with foreach and non-finite checks |
| Language-model loss | Project's stable cross-entropy | `torch.nn.functional.cross_entropy` over flattened token positions |
| Model parameter layout | Original model | Kept aligned with the original model |

## Benchmark scope

The technical report's RTX 3070 Ti short benchmark measured cumulative PyTorch-native replacements at 153.71 ms per step (213,186 tokens/s), compared with 236.37 ms per step (138,630 tokens/s) for its original baseline. Peak tensor memory was 3.79 GiB versus 6.57 GiB. That is a 1.54× throughput ratio and about 42% lower peak tensor memory for batch 128 × sequence 256, PyTorch 2.14.0+cu126.

These are short-run measurements, not a 10,000-step convergence comparison. Runtime depends on GPU, PyTorch/CUDA build, input pipeline, and compile behavior. Optimizer checkpoint migration between the snapshots is not validated; use a fresh run when switching implementations.

## Deliberately not enabled by default

The report's Liger fused linear-cross-entropy candidate was not installed or measured. Its KV-cache prototype passed numerical comparisons but did not show stable extra generation speed over SDPA plus final-position logits on the measured GPU. Neither is presented as an established optimization in the PyTorch-native version.
