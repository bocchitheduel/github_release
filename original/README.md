# Original educational implementation

This snapshot keeps the project's custom tiled attention, AdamW, gradient clipping, and cross-entropy. It also includes an optional Muon + AdamW optimizer. On CUDA FP32, Muon groups matrices by shape and uses batched Newton–Schulz kernels; `block_attention_triton.py` is supplied for direct attention-kernel experiments and is not the model's default path.

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

The model has 4 decoder layers, 16 attention heads, width 512, SwiGLU width 1,344, RoPE, RMSNorm, and separate embedding/LM-head weights. See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) and `reports/` for the architecture, the AdamW/Muon training comparison, and kernel assessments.

The recorded Muon + AdamW run reached validation loss 1.324593 at 10,000 steps, versus 1.376768 for a historical AdamW run. This is a single-seed comparison, and the old baseline did not retain a dataset hash. The PyTorch-native operator short benchmark measured 1.54× throughput and 42% lower peak tensor allocation on an RTX 3070 Ti, but it did not include a new 10,000-step run.
