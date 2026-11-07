# Original educational implementation

This snapshot retains the project’s custom attention, cross-entropy, gradient clipping, and AdamW code. The model’s library SDPA option is available for inference, while the educational block implementation remains the training model default.

## Requirements

Python 3.11 or newer; PyTorch 2.4 or newer; NumPy; `regex`. PyYAML is needed only for YAML-based training configurations. Gradio is needed only for `web_ui.py`. Install the core dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install optional UI/config dependencies with `python -m pip install gradio PyYAML`. For CUDA, install a PyTorch wheel matching your CUDA runtime.

## Train

Provide raw binary token files matching the report's `int64` format, then run:

```bash
python train_loop.py \
  --train-path /path/to/train.bin \
  --val-path /path/to/val.bin \
  --vocab-size 8983 \
  --device auto
```

The defaults use the report's generic training configuration. `train_10k.py` describes the fixed 128 × 256, 10,000-step experiment and expects the corresponding data files locally. No dataset or checkpoint is bundled.

## Architecture and results

The model is a 4-layer decoder-only Transformer with 16 attention heads, RoPE, RMSNorm, SwiGLU, and separate embedding/LM-head weights. See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) and the included `assessments/` artifacts for architecture, training history, and benchmark methodology.

The report's external-operator short benchmark measured 1.54× throughput and 42% lower peak tensor memory on an RTX 3070 Ti, PyTorch 2.14.0+cu126, batch 128 × sequence 256. It did not include a new 10,000-step convergence run; these figures are hardware- and configuration-specific.
