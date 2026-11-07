"""Read-only checks of generation contracts used by the inference report."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from bpe import BPE
from generate import generate, sample_top_p
from transformer_LM import TransformerLM

torch.set_num_threads(2)
torch.manual_seed(123)
vocab = {i: bytes([i]) for i in range(256)}
vocab[256] = b'<S>'
tokenizer = BPE(vocab, [], {'<S>':256})
model = TransformerLM(257,16,num_layers=1,num_heads=4,d_ffn=32,max_seq_len=4)
checks = []

seen = []
forward = model.forward_last_logits
def traced(ids, *args, **kwargs):
    seen.append(ids[0].tolist())
    return forward(ids,*args,**kwargs)
with patch.object(model,'forward_last_logits',side_effect=traced), \
     patch('generate.sample_top_p',side_effect=[33,33,256]):
    output=generate(model,tokenizer,'abcdef',max_new_tokens=3,amp=False)
assert seen == [list(b'cdef'),list(b'def!'),list(b'ef!!')]
assert output == 'abcdef!!'
checks.append(dict(name='context_crop_rollover_and_eos',passed=True,contexts=seen,output=output))

model.train()
with patch('generate.sample_top_p',side_effect=RuntimeError('injected sampling failure')):
    try:
        generate(model,tokenizer,'ab',max_new_tokens=1,amp=False)
    except RuntimeError as error:
        assert str(error)=='injected sampling failure'
    else:
        raise AssertionError('expected error')
assert model.training
checks.append(dict(name='exception_restores_model_mode',passed=True))

state=torch.get_rng_state().clone()
a=generate(model,tokenizer,'ab',max_new_tokens=8,seed=42,amp=False)
b=generate(model,tokenizer,'ab',max_new_tokens=8,seed=42,amp=False)
assert a==b and torch.equal(state,torch.get_rng_state())
checks.append(dict(name='seeded_repeatability_without_global_rng_consumption',passed=True))

with patch.object(model,'forward_last_logits') as mocked:
    output=generate(model,tokenizer,'ab',max_new_tokens=0,amp=False)
    mocked.assert_not_called()
assert output=='ab'
checks.append(dict(name='zero_new_tokens_skips_model',passed=True))

for kwargs in [dict(prompt=''),dict(prompt='ab',temperature=0),
               dict(prompt='ab',top_p=0),dict(prompt='ab',top_p=1.1),
               dict(prompt='ab',max_new_tokens=-1)]:
    try:
        generate(model,tokenizer,amp=False,**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(kwargs)
checks.append(dict(name='invalid_inputs_rejected',passed=True,cases=5))

logits=torch.tensor([0.6,0.3,0.1]).log()
rng=torch.Generator().manual_seed(42)
counts=[0,0,0]
for _ in range(4096):
    counts[sample_top_p(logits,top_p=0.8,generator=rng)]+=1
assert counts[2]==0 and abs(counts[0]/4096-2/3)<0.04
checks.append(dict(name='top_p_crossing_token_and_renormalization',passed=True,
                   counts=counts,expected_probabilities=[2/3,1/3,0]))

result=dict(device='cpu',checks=checks,passed=len(checks))
(Path(__file__).parent/'contracts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
