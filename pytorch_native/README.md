# PyTorch-native optimized implementation

This snapshot defaults to PyTorch SDPA, uses `torch.optim.AdamW` with fused CUDA kernels when supported, uses PyTorch foreach gradient clipping, and uses `torch.nn.functional.cross_entropy`. The model architecture, activation checkpointing, Muon options, and parameter layout are kept aligned with `original/`; the block attention option remains available for comparison.

## Requirements

Python 3.11 or newer; PyTorch 2.4 or newer; NumPy; and `regex`. PyYAML is needed for YAML training configurations, Gradio for `web_ui.py`, and a Triton build compatible with the installed PyTorch/CUDA stack for the CUDA Muon fast path and standalone Triton attention experiments. Install the core dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install optional UI/config dependencies with `python -m pip install gradio PyYAML`. For CUDA, install a PyTorch wheel matching your CUDA runtime.

## Train

Provide raw binary token-ID files in little-endian `int64` format, then run:

```bash
python train_loop.py \
  --train-path /path/to/train.bin \
  --val-path /path/to/val.bin \
  --vocab-size 8983 \
  --device auto
```

`train_10k.py` describes the fixed batch 128 × sequence 256, 10,000-step experiment. No dataset or model checkpoint is bundled.

## Architecture and results

The model has 4 decoder layers, 16 attention heads, width 512, SwiGLU width 1,344, RoPE, RMSNorm, and separate embedding/LM-head weights. See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md), `reports/`, and [`../COMPARISON.md`](../COMPARISON.md) for architecture, implementation differences, and measured results.

The short RTX 3070 Ti operator benchmark measured 1.54× throughput and about 42% lower peak tensor allocation than the original baseline. It did not include a 10,000-step convergence run. The Muon training comparison is a separate single-seed result and is not evidence of a PyTorch-native full-run comparison. Cross-version optimizer checkpoint restoration has not been validated; start a fresh run when switching optimizer implementations.
