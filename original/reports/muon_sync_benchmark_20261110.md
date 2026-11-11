# Muon CUDA synchronization benchmark

- Date: 2026-11-10T18:14:47+08:00
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU; PyTorch 2.14.0+cu126; CUDA 12.6
- Workload: 8 FP32 matrices, 5 Newton–Schulz steps, 20 measured steps per case; TF32 disabled.
- `natural` keeps Muon's existing finite-gradient check; `step` adds `torch.cuda.synchronize()` after every `Muon.step()`; `kernel` adds a barrier after each Muon Triton wrapper and `torch.bmm`.
- CPU work is a synthetic fixed-duration host task between steps. `0 ms` isolates synchronization; the second setting tests whether queued GPU work overlaps host work.

| Extra sync mode | Host work ms/step | Wall ms/step | CUDA event timeline ms/step | Wall slowdown vs natural |
|---|---:|---:|---:|---:|
| Existing sync only | 0.000 | 35.558 | 35.533 | 1.000× |
| After every step | 0.000 | 35.985 | 35.965 | 1.012× |
| After each kernel | 0.000 | 44.666 | 44.630 | 1.256× |
| Existing sync only | 0.500 | 35.711 | 35.693 | 1.000× |
| After every step | 0.500 | 37.769 | 37.743 | 1.058× |
| After each kernel | 0.500 | 46.499 | 46.471 | 1.302× |

## Existing synchronization point

The current batched CUDA Muon path reduces finite-gradient flags to a Python bool in `_step_batched_cuda`. That conversion synchronizes the CPU with CUDA once per `Muon.step()`. `natural` therefore means no additional barriers; it is not fully asynchronous.

## Interpretation

The after-step case measures the cost of waiting for the update kernels before returning from `step()`. The after-kernel case is intentionally diagnostic and inserts many barriers inside Newton–Schulz. Results are synthetic optimizer timings, not end-to-end training throughput.
