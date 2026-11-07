"""Assess the current generator and isolated inference optimizations.

Synthetic timing trajectories use identical real validation IDs to control
sequence length independently of EOS. Public-API samples retain normal EOS
behavior. This historical CUDA benchmark requires a CUDA device.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import generate as generation

ORIGINAL_SAMPLE = generation.sample_top_p


def emit(stream, **row):
    line = json.dumps(row, ensure_ascii=False)
    print(line, flush=True)
    stream.write(line + '\n')
    stream.flush()


class Runner:
    def __init__(self, model, variant, cache_dtype=torch.bfloat16):
        self.model, self.variant = model, variant
        self.length = 0
        self.limit = model.blocks[0].attention.max_seq_len
        self.cache = []
        if variant == 'sdpa_kv_cache':
            for block in model.blocks:
                a = block.attention
                shape = (1, a.num_heads, self.limit, a.head_dim)
                self.cache.append(tuple(torch.empty(shape, device='cuda', dtype=cache_dtype)
                                        for _ in range(2)))

    def activate(self):
        backend = 'sdpa' if self.variant.startswith('sdpa') else 'block'
        for block in self.model.blocks:
            block.attention.attention_backend = backend

    def reset(self):
        self.length = 0

    @torch.inference_mode()
    def __call__(self, ids):
        if self.variant == 'current':
            return self.model(ids)[0, -1]
        if self.variant != 'sdpa_kv_cache':
            x = self.model.embedding(ids)
            for block in self.model.blocks:
                x = block(x)
            return self.model.lm_head(self.model.final_norm(x[:, -1:]))[0, -1]
        # Preserve current sliding-window semantics by rebuilding at overflow.
        incremental = self.length > 0 and ids.shape[1] == self.length + 1
        offset = self.length if incremental else 0
        tokens = ids[:, -1:] if incremental else ids
        x = self.model.embedding(tokens)
        total = offset + tokens.shape[1]
        for index, block in enumerate(self.model.blocks):
            a = block.attention
            normalized = block.attn_norm(x)
            def heads(project):
                return project(normalized).reshape(1, tokens.shape[1], a.num_heads,
                                                    a.head_dim).transpose(1, 2)
            q = a.rope(heads(a.q_proj), offset=offset)
            k = a.rope(heads(a.k_proj), offset=offset)
            v = heads(a.v_proj)
            kc, vc = self.cache[index]
            kc[..., offset:total, :].copy_(k)
            vc[..., offset:total, :].copy_(v)
            attended = F.scaled_dot_product_attention(
                q, kc[..., :total, :], vc[..., :total, :],
                is_causal=not incremental, dropout_p=0.0)
            combined = attended.transpose(1, 2).reshape(1, tokens.shape[1], a.d_model)
            x = x + a.out_proj(combined)
            x = x + block.ffn(block.ffn_norm(x))
        self.length = total
        return self.model.lm_head(self.model.final_norm(x[:, -1:]))[0, -1]


def sync():
    torch.cuda.synchronize()


def measured_prefill(runner, ids):
    runner.activate()
    times = []
    for index in range(7):
        runner.reset()
        sync()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            runner(ids)
        sync()
        if index >= 2:
            times.append(1000*(time.perf_counter()-start))
    return statistics.median(times)


def measured_decode(runner, data, length, count=16):
    runner.activate()
    samples = []
    peaks = []
    for repetition in range(4):
        runner.reset()
        context = data[:length].reshape(1, -1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            runner(context)
        sync()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for step in range(count):
            context = torch.cat((context, data[length+step].reshape(1, 1)), dim=1)[:, -512:]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                runner(context)
            # Real generate() synchronizes on sample_top_p().item() each step.
            sync()
        elapsed = time.perf_counter()-start
        if repetition:
            samples.append(1000*elapsed/count)
            peaks.append(torch.cuda.max_memory_allocated()/2**20)
    ms = statistics.median(samples)
    return dict(decode_ms_per_token=ms, decode_tokens_per_second=1000/ms,
                decode_rounds_ms_per_token=samples,
                peak_allocated_mib=max(peaks))


def compare(a, b):
    delta = a.float()-b.float()
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm()/a.float().norm()).item(),
                same_argmax=a.argmax().item()==b.argmax().item())


def public_sample(model, tokenizer, prompt, max_new_tokens=128):
    sampled = []
    first = None
    start = None
    def traced(logits, temperature=1.0, top_p=0.8, generator=None):
        nonlocal first
        token = ORIGINAL_SAMPLE(logits, temperature, top_p, generator)
        sampled.append(token)
        if first is None:
            first = time.perf_counter()
        return token
    for block in model.blocks:
        block.attention.attention_backend = 'block'
    generation.sample_top_p = traced
    sync()
    torch.cuda.reset_peak_memory_stats()
    try:
        start = time.perf_counter()
        output = generation.generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                                     temperature=0.8, top_p=0.9, seed=42, amp=True)
        sync()
        seconds = time.perf_counter()-start
    finally:
        generation.sample_top_p = ORIGINAL_SAMPLE
    special = set(tokenizer.special_tokens.values())
    new_count = sum(x not in special for x in sampled)
    stop = bool(sampled and sampled[-1] in special)
    return dict(prompt=prompt, prompt_tokens=len(tokenizer.encode(prompt)),
                max_new_tokens=max_new_tokens, generated_tokens=new_count,
                sampled_steps=len(sampled), stopped_on_special=stop,
                ttft_ms=1000*(first-start) if first else None,
                total_seconds=seconds, generated_tokens_per_second=new_count/seconds,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                generated_token_ids=sampled, text=output)


def export_snapshot():
    rows=[json.loads(line) for line in (OUT/'results.jsonl').read_text().splitlines()]
    summary=dict(environment=next(r for r in rows if r['event']=='environment'),
                 load=next(r for r in rows if r['event']=='model_load'),
                 controlled={},requests=[],cache_validation={},
                 sampling=next(r for r in rows if r['event']=='sampling'))
    requests=[r for r in rows if r['event']=='public_generate']
    for row in rows:
        if row['event']=='controlled_inference':
            summary['controlled'].setdefault(str(row['prompt_tokens']),{})[row['variant']]={
                k:row[k] for k in ('prefill_ms','decode_ms_per_token',
                                   'decode_tokens_per_second','peak_allocated_mib')}
    summary['requests']=[{k:v for k,v in row.items() if k not in ('text','generated_token_ids')}
                         for row in requests]
    for precision,event in [('bf16','cache_numerics'),('fp32','cache_numerics_fp32')]:
        values=[r for r in rows if r['event']==event]
        summary['cache_validation'][precision]=dict(
            max_absolute_error=max(r['max_abs'] for r in values),
            max_relative_l2=max(r['relative_l2'] for r in values),
            all_same_argmax=all(r['same_argmax'] for r in values),
            history_lengths=[r['history_tokens'] for r in values])
    for values in summary['controlled'].values():
        baseline=values['current']['decode_ms_per_token']
        for value in values.values():
            value['speedup_vs_current']=baseline/value['decode_ms_per_token']
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    parts=['# 推理生成样例','',
           '参数：现有训练检查点、原始 generate()、BF16、temperature=0.8、top_p=0.9、seed=42、最多 128 个新 token。每例同 seed 重复生成的 token 序列一致。以下四例仅作定性观察。','']
    for index,row in enumerate(requests,1):
        parts.extend([f'## 样例 {index}','',f'Prompt：{row["prompt"]}','',
                      f'输入 {row["prompt_tokens"]} token；生成 {row["generated_tokens"]} token；特殊 token 早停：{row["stopped_on_special"]}。重复请求耗时 {row["repeat_total_seconds"]:.3f} 秒，首 token {row["repeat_ttft_ms"]:.2f} ms。',
                      '','```text',row['text'],'```',''])
    (OUT/'samples.md').write_text('\n'.join(parts),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(ROOT/'runs/b128_s256_10k_20261015_171649_473135/checkpoints/latest.pt'))
    parser.add_argument('--tokenizer', default='tokenizer_julia/models/sample_model.toml')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this inference benchmark')
    with (OUT/'results.jsonl').open('w') as stream:
        start = time.perf_counter()
        torch.cuda.init()
        sync()
        init_seconds = time.perf_counter()-start
        emit(stream, event='environment', gpu=torch.cuda.get_device_name(0),
             torch=torch.__version__, device_init_seconds=init_seconds,
             tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
             compiled=False, amp='bfloat16', batch_size=1,
             methodology='prefill median of 5; decode median of 3x16 after warmup; per-step CUDA sync')
        start = time.perf_counter()
        model, tokenizer = generation.load_model(
            args.checkpoint, args.tokenizer, 'cuda', attention_backend='block')
        sync()
        emit(stream, event='model_load', seconds=time.perf_counter()-start,
             checkpoint_bytes=Path(args.checkpoint).stat().st_size,
             parameter_mib=sum(p.numel()*p.element_size() for p in model.parameters())/2**20,
             allocated_mib=torch.cuda.memory_allocated()/2**20)
        data_np = np.memmap(ROOT/'val.bin',dtype=np.int64,mode='r')
        data = torch.tensor(np.array(data_np[:1024]), device='cuda', dtype=torch.long)
        # All timing variants share the same loaded trained model and token IDs.
        for variant in ['current','last_head','sdpa_last_head','sdpa_kv_cache']:
            runner = Runner(model, variant)
            runner.activate()
            manager = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if variant.startswith('sdpa') else nullcontext()
            with manager:
                for length in [16,128,256,480,512]:
                    ids = data[:length].reshape(1,-1)
                    prefill = measured_prefill(runner, ids)
                    emit(stream,event='controlled_inference',variant=variant,
                         prompt_tokens=length,decode_steps=16,prefill_ms=prefill,
                         **measured_decode(runner,data,length))
            del runner
        # Numerical comparison at the cache boundary and after window overflow.
        cached = Runner(model, 'sdpa_kv_cache')
        reference = Runner(model, 'sdpa_last_head')
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for length in range(496,515):
                ids = data[max(0,length-512):length].reshape(1,-1)
                cached.activate()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    cached_out=cached(ids)
                    if length in (496,497,511,512,513,514):
                        reference.activate()
                        full_out=reference(ids)
                        result=compare(full_out,cached_out)
                        assert result['relative_l2']<0.03,result
                        emit(stream,event='cache_numerics',history_tokens=length,
                             context_tokens=ids.shape[1],**result)
        del cached,reference
        # FP32 removes most precision ambiguity when checking cache semantics.
        cached=Runner(model,'sdpa_kv_cache',cache_dtype=torch.float32)
        reference=Runner(model,'sdpa_last_head')
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for length in range(508,515):
                ids=data[max(0,length-512):length].reshape(1,-1)
                cached.activate()
                cached_out=cached(ids)
                if length in (508,509,512,513,514):
                    reference.activate()
                    result=compare(reference(ids),cached_out)
                    assert result['relative_l2']<5e-5,result
                    emit(stream,event='cache_numerics_fp32',history_tokens=length,
                         context_tokens=ids.shape[1],**result)
        del cached,reference
        for length in (16,256,512):
            ids=data[:length].reshape(1,-1)
            current=Runner(model,'current')
            last=Runner(model,'last_head')
            optimized=Runner(model,'sdpa_last_head')
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                current.activate()
                baseline=current(ids)
                last.activate()
                head_result=compare(baseline,last(ids))
                optimized.activate()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    sdpa_result=compare(baseline,optimized(ids))
            emit(stream,event='backend_numerics',context_tokens=length,
                 last_head=head_result,sdpa_last_head=sdpa_result)
        # Keep normal EOS stopping and actual token counts for real requests.
        for prompt in ['Once upon a time',
                       'Lily found a small red box in the garden. Inside the box',
                       'Explain why the sky is blue.',
                       '请介绍一下你自己。']:
            first=public_sample(model,tokenizer,prompt)
            second=public_sample(model,tokenizer,prompt)
            assert first['generated_token_ids']==second['generated_token_ids']
            emit(stream,event='public_generate',seed=42,temperature=0.8,top_p=0.9,
                 repeated_seed_identical=True,repeat_total_seconds=second['total_seconds'],
                 repeat_ttft_ms=second['ttft_ms'],
                 repeat_generated_tokens_per_second=second['generated_tokens_per_second'],**first)
        # Isolate CPU/GPU sampling and synchronization cost at the real vocab size.
        logits=torch.linspace(-3,3,model.vocab_size,device='cuda')
        rng=torch.Generator(device='cuda').manual_seed(42)
        for _ in range(10):
            ORIGINAL_SAMPLE(logits,0.8,0.9,rng)
        start=time.perf_counter()
        for _ in range(100):
            ORIGINAL_SAMPLE(logits,0.8,0.9,rng)
        emit(stream,event='sampling',vocab_size=model.vocab_size,
             ms_per_sample=1000*(time.perf_counter()-start)/100)
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                ['generate.py','web_ui.py','MH_selfattention.py','block_attention.py',
                 'transformer_LM.py','RoPE_module.py']}
        (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    export_snapshot()


if __name__=='__main__':
    main()
