# Muon optimizer step: kernel time and memory

Date: 2026-11-10

## Setup

- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8 GiB
- PyTorch 2.14.0, CUDA 12.6
- Model dimensions from `configs/muon_rms_cosine_2000.yaml`: vocabulary 8,983, width 512, 4 layers, FFN width 1,344
- Muon: FP32 parameters, gradients, momentum, and Newton–Schulz math; 5 NS iterations; Nesterov enabled
- Each benchmark step uses preallocated random gradients. Model forward/backward is excluded.
- Timings are CUDA-event and synchronized wall-clock measurements after 3 warmup steps; steady-state statistics use 20 steps. CUDA profiler attribution is from one additional instrumented step.
- Memory figures use MiB (2^20 bytes) and PyTorch allocator counters. They exclude CUDA context memory and training activations.

## Baseline Muon-only result

The model has 28 Muon matrices: 16 of shape 512×512, 8 of 1,344×512, and 4 of 512×1,344. That is 12,451,840 parameters, or 47.5 MiB in FP32. The momentum state adds another 47.5 MiB on the first update.

| Measurement | Result |
|---|---:|
| Steady step, synchronized wall time (median) | 38.8–38.9 ms |
| Steady step, CUDA-event time (median) | 38.5–38.7 ms |
| Wall-time mean across 20 steps | 39.0–40.0 ms |
| Wall-time p90 across runs | 40.8–43.0 ms |
| CUDA-profiler kernel time for one step | 30.6 ms |
| Actual CUDA kernel launches in that profiled step | 1,911 |

The 5 NS iterations perform 3 matrix multiplications per iteration for each matrix: 28 × 5 × 3 = 420 GEMMs. The profiler attributed 22.83 ms to these GEMMs (about 75% of kernel time). The remaining approximately 7.77 ms is elementwise arithmetic, reductions, gradient-finiteness checks, and parameter updates.

| CUDA kernel family | Calls | Total time |
|---|---:|---:|
| `ampere_sgemm_64x32_sliced1x4_nn` | 220 | 8.79 ms |
| `ampere_sgemm_128x64_tn` | 60 | 5.24 ms |
| `ampere_sgemm_128x64_nt` | 40 | 3.49 ms |
| `ampere_sgemm_32x128_tn` | 80 | 3.49 ms |
| `ampere_sgemm_64x64_nn` | 20 | 1.70 ms |
| Vectorized multiply kernels | 476 | 2.87 ms |
| Vectorized add kernels | 352 | 1.94 ms |

The profiled kernel bodies total about 30.6 ms, while the steady CUDA-event interval is about 38.5 ms. The gap is consistent with eager Python dispatch and GPU launch gaps between many small operations; it is not a direct measurement of CPU time alone.

| Muon-only memory counter | Result |
|---|---:|
| Allocated before first optimizer step (model plus Muon gradients) | 131.25 MiB |
| First-step peak allocated | 205.65 MiB |
| Steady-state allocated after step | 186.89 MiB |
| Steady-state peak allocated | 205.65 MiB |
| Steady-state peak above post-step allocation | 18.75 MiB |
| Peak reserved by PyTorch allocator | 230.00 MiB |

The first update is cold and includes lazy state allocation and CUDA library initialization; it measured 183–213 ms across two fresh-process runs. It is not representative of later steps.

## Baseline Muon + AdamW bundle

The configured model assigns 28 matrix tensors to Muon and 11 tensors (embedding, output projection, and normalization parameters) to AdamW. With all model gradients preallocated, the full bundle measured a 41.0 ms steady-step median. It held 117.71 MiB of optimizer state (47.5 MiB Muon momentum plus AdamW moments).

| Bundle memory counter | Result |
|---|---:|
| Model parameters | 82.61 MiB |
| All gradients | 82.61 MiB |
| Optimizer state after initialization | 117.71 MiB |
| Allocated after steady step | 294.49 MiB |
| Steady-state peak allocated during optimizer step | 313.24 MiB |
| First-step peak allocated | 312.49 MiB |

These are optimizer-update-only figures. Full training also retains forward/backward activations and other workspaces, which are not measured here.

## Batched CUDA implementation

The CUDA FP32 path now groups matrices by their Newton–Schulz orientation. For this model, that gives a batch of 16 matrices shaped 512×512 and a batch of 12 shaped 512×1,344. A Triton kernel fuses momentum, Nesterov, and partial norm work; batched GEMMs use `baddbmm` for the polynomial and X-update epilogues; a Triton kernel fuses scale, weight decay, and parameter update. The existing matrix-at-a-time path remains for CPU and non-FP32 parameters or gradients.

| Muon-only steady-state measurement | Before | Batched CUDA | Change |
|---|---:|---:|---:|
| Synchronized wall time, median | 38.77 ms | 26.28 ms | 32.3% lower latency (1.48× throughput) |
| CUDA-event time, median | 38.50 ms | 26.08 ms | 32.3% lower |
| CUDA kernel launches per profiled step | 1,911 | 72 | 96.2% fewer |
| Profiler kernel-body time | 30.60 ms | 21.82 ms | 28.7% lower |
| Peak allocated memory | 205.64 MiB | 388.01 MiB | +182.37 MiB |
| Peak reserved by PyTorch allocator | 230.00 MiB | 404.00 MiB | +174.00 MiB |

The optimized step issues 10 batched Gram GEMMs and 20 batched `baddbmm` calls for 5 NS iterations across 2 shape groups. The profile also shows 20 device-to-device copies used to initialize the `baddbmm` epilogues.

The full Muon + AdamW bundle measured 28.78 ms median per steady update, down from 41.05 ms (29.9% lower latency). Its optimizer-only peak allocated memory was 514.05 MiB, up from 313.24 MiB; allocation after the step was 496.05 MiB. The optimized path keeps its workspaces for reuse, which accounts for most of the added memory.

The cold first Muon update took about 433 ms in the final run, including Triton/CUDA initialization. It is not representative of steady-state timing.

## Correctness checks

- Compared two updates against the matrix-at-a-time implementation on all 28 configured matrix shapes: maximum parameter difference 2.38e-7; maximum momentum difference 4.77e-7.
- Checked partial gradients, non-finite gradient rejection before state mutation, and checkpoint momentum restoration.
- `PYTHONPATH=$PWD .venv/bin/pytest -q tests/test_muon.py`: 9 passed, 5 subtests passed.

## Takeaways

1. Batching and fusion reduced steady Muon latency by about one third and lowered the profiler's kernel count from roughly 1,900 to 72.
2. The speedup costs about 182 MiB more peak allocated memory for Muon-only updates. The current benchmark remains well below the 8 GiB device limit, but this workspace cost should be included when budgeting full-training activation memory.
3. Reducing the configured NS iteration count from 5 would further reduce GEMM work linearly, with a corresponding change in orthogonalization quality.
4. The project compiles the model and loss; the optimizer uses its own Triton/CUDA fast path rather than being captured by `torch.compile`.
