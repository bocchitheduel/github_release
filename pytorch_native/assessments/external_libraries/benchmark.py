"""Compare training backends without changing the assignment implementation.

Uses the existing 128 x 256 configuration and real token batches. No model or
optimizer checkpoints are saved. Run each variant in its own process so CUDA
memory and compilation state do not carry between measurements.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(OUT / 'cache/inductor'))
os.environ.setdefault('TRITON_CACHE_DIR', str(OUT / 'cache/triton'))


def emit(**values):
    print(json.dumps(values), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['current', 'sdpa', 'sdpa_fused', 'sdpa_fused_ce', 'suite'])
    args = parser.parse_args()
    if args.variant == 'suite':
        for variant in ['current', 'sdpa', 'sdpa_fused', 'sdpa_fused_ce']:
            with (OUT / f'{variant}.jsonl').open('w') as stream, (OUT / f'{variant}.stderr').open('w') as errors:
                emit(event='starting_process', variant=variant)
                result = subprocess.run([sys.executable, '-u', __file__, variant], stdout=stream, stderr=errors)
                emit(event='finished_process', variant=variant, returncode=result.returncode)
        return

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from adamW import AdamW
    from crossentropy import cross_entropy
    from dataloader import DataLoader
    from train_utils import clip_grad_norm_
    from transformer_LM import TransformerLM
    import MH_selfattention

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)
    emit(event='environment', gpu=torch.cuda.get_device_name(0), torch=torch.__version__, variant=args.variant)

    def sdpa(q, k, v, q_block_size=64, kv_block_size=64, *, attn_mask=None):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=True, dropout_p=0.0)

    if args.variant != 'current':
        MH_selfattention.block_causal_attention = sdpa
    model = TransformerLM(vocab_size=8983, d_model=512, num_layers=4, num_heads=16,
                          d_ffn=1344, max_seq_len=512, device='cuda')
    fused = args.variant in ('sdpa_fused', 'sdpa_fused_ce')
    optimizer = (torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
                 if fused else AdamW(model.parameters(), lr=1e-3))
    clip = (lambda p: torch.nn.utils.clip_grad_norm_(p, 1.0, foreach=True, error_if_nonfinite=True)) if fused else (lambda p: clip_grad_norm_(p, 1.0))
    loss_fn = ((lambda logits, targets: F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
               if args.variant == 'sdpa_fused_ce' else cross_entropy)
    loss_fn = torch.compile(loss_fn)
    model.compile()
    loader = DataLoader(ROOT / 'train.bin', 128, 256, device='cuda', seed=42)
    inputs, targets = next(loader)
    emit(event='starting_compile', parameters=sum(p.numel() for p in model.parameters()))

    def step(load=False):
        nonlocal inputs, targets
        if load:
            inputs, targets = next(loader)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = loss_fn(model(inputs), targets)
        loss.backward()
        clip(model.parameters())
        optimizer.step()

    def timing(fn, warmup=10, repeats=10, rounds=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
            samples.append(1000 * (time.perf_counter() - start) / repeats)
        ms = statistics.median(samples)
        return dict(ms=ms, samples_ms=samples, tokens_per_second=32768000/ms,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        start = time.perf_counter()
        step()
        torch.cuda.synchronize()
        emit(event='first_step_complete', seconds=time.perf_counter()-start)
        emit(event='fixed_batch_train_step', **timing(step))
        # Touch the same sampled file pages before every variant. This avoids
        # giving later variants an unfair advantage from the OS page cache.
        import copy
        data_rng_state = copy.deepcopy(loader.rng.bit_generator.state)
        for _ in range(80):
            next(loader)
        torch.cuda.synchronize()
        loader.rng.bit_generator.state = data_rng_state
        emit(event='train_step_with_loader', data_pages_prewarmed=True,
             **timing(lambda: step(True)))
        if args.variant == 'sdpa_fused':
            emit(event='dataloader', **timing(lambda: next(loader), repeats=10))


if __name__ == '__main__':
    main()
