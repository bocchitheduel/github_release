# CS336 Language Model Implementations

This repository contains two runnable source snapshots based on the accompanying technical report:

- [`original/`](original/): the educational implementation, including custom block attention, optimizer, clipping, and cross-entropy implementations.
- [`pytorch_native/`](pytorch_native/): the same model and training workflow with the measured PyTorch-native training operators enabled by default.

The external-library version uses PyTorch SDPA, fused `torch.optim.AdamW` when CUDA is available, PyTorch foreach gradient clipping, and `torch.nn.functional.cross_entropy`. The custom implementation remains available in `original/` for study and comparison. The report's Liger kernel and KV-cache notes remain evaluation candidates, not default-path claims.

See [`COMPARISON.md`](COMPARISON.md) for the exact operator changes and benchmark scope. The technical report records the original project and measured prototypes; it does not claim a new 10,000-step training run for the PyTorch-native snapshot.

## Data and checkpoints

Datasets, tokenized `.bin` files, checkpoints, and compiler caches are intentionally excluded. Supply your own token files and checkpoint paths. The technical report and compact assessment artifacts are included.

## Start here

Each directory is an independent project snapshot with its own README, dependencies, and source files. Follow that directory's setup and CLI instructions.
