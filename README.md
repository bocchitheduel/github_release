# CS336 Assignment 1 Language Models

This repository contains two runnable versions of a 21.66M-parameter TinyStories language model. Both snapshots now include shared model and training updates through 2026-11-11:

- [`original/`](original/): educational PyTorch implementations of tiled attention, AdamW, gradient clipping, and cross-entropy, with optional Muon training.
- [`pytorch_native/`](pytorch_native/): PyTorch SDPA, fused AdamW on CUDA, foreach gradient clipping, and PyTorch cross-entropy as the default training operators. The educational attention backend remains selectable.

The model is a 4-layer decoder Transformer with 16 heads, width 512, SwiGLU width 1,344, RoPE, RMSNorm, an 8,983-token byte-level BPE vocabulary, and separate input/output embeddings. The technical report in each snapshot documents architecture, training setup, inference limits, and measured results.

## Results and evidence

A single RMS-aligned Muon + AdamW run reached validation loss **1.324593** at 10,000 steps, compared with **1.376768** in the historical AdamW run. This is a useful single-seed comparison; the older AdamW run did not preserve dataset hashes, so byte-for-byte data identity cannot be confirmed. It is not evidence of a general optimizer advantage.

A short RTX 3070 Ti benchmark of the PyTorch-native operator bundle measured 1.54× throughput and 42% lower peak tensor allocation than the original baseline. A separate Triton attention kernel benchmark was close to FlashAttention 2 on the tested shapes. These are operator measurements, not a new 10,000-step convergence run; the Triton attention kernel is not wired into the model's default attention path.

See [`COMPARISON.md`](COMPARISON.md) for version differences and benchmark scope, and [`original/TECHNICAL_REPORT.md`](original/TECHNICAL_REPORT.md) for the full report.

## Data and checkpoints

Datasets, tokenized `.bin` files, model checkpoints, and compiler caches are excluded. Provide token files in little-endian `int64` format and a matching tokenizer. Optimizer checkpoint states are implementation-specific; start a fresh run when switching optimizer implementations unless you have validated the migration.

## Start here

Each directory has its own README, requirements, configs, source files, and assessment artifacts. Follow that directory's setup instructions.
