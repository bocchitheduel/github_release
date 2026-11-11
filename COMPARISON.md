# Version comparison

| Area | `original/` | `pytorch_native/` |
| --- | --- | --- |
| Default causal attention | Project tiled attention with online softmax and a manual first-order backward | PyTorch SDPA; `--attention-backend block` selects the tiled implementation |
| Alternative Triton attention | `block_attention_triton.py` is included for direct kernel experiments; it is not the model default | Same standalone kernel and benchmark artifacts; not selected by the model |
| Default AdamW | Project implementation | `torch.optim.AdamW`, fused on CUDA when supported |
| Muon | Optional hybrid Muon + AdamW; CUDA FP32 matrices use batched Newton–Schulz kernels | Same Muon matrix path; PyTorch AdamW updates the remaining parameters |
| Gradient clipping | Project implementation | PyTorch `clip_grad_norm_` with foreach and non-finite checks |
| Language-model loss | Project stable cross-entropy | `torch.nn.functional.cross_entropy` over flattened token positions |
| Activation checkpointing | Blocks are checkpointed in groups of two during training by default | Same model behavior |
| Parameter layout | Separate embedding and LM-head weights | Same parameter layout |

## Benchmark scope

The short RTX 3070 Ti training benchmark measured the cumulative PyTorch-native operator bundle at 153.71 ms per step (213,186 tokens/s), compared with 236.37 ms per step (138,630 tokens/s) for its original baseline. Peak tensor allocation was 3.79 GiB versus 6.57 GiB: 1.54× throughput and about 42% lower peak allocation for batch 128 × sequence 256, PyTorch 2.14.0+cu126.

These are short-run measurements, not a full convergence comparison. A separate single-seed Muon + AdamW run reached validation loss 1.324593 at 10,000 steps; the historical AdamW baseline reached 1.376768. The baseline did not record dataset hashes, and the Muon result should be reproduced from a common initialization before drawing a general optimizer conclusion.

The standalone Triton attention benchmark used BF16, batch 1, 8 heads, and head dimension 64 on an RTX 3070 Ti. It was faster than the prior local Triton snapshot by 1.4%–7.5% in forward + backward over lengths 128–2048, and near FlashAttention 2.8.3. At length 2048 it was about 1% slower than FA2. These results do not describe the full model training path.

Runtime and memory depend on hardware, PyTorch/CUDA build, input pipeline, and compilation. Optimizer checkpoint migration between implementations has not been validated.

## Optional paths

The Liger fused linear-cross-entropy path is optional and is not the default. The KV-cache prototype passed numerical checks but showed no stable extra generation speed over SDPA plus final-position logits in the measured workload.
