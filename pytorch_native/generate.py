"""Autoregressive text generation with temperature and nucleus sampling."""
import argparse
from codecs import getincrementaldecoder
from dataclasses import dataclass
import time
from typing import Iterator

import torch

from bpe import BPE
from transformer_LM import TransformerLM
from amp_utils import bf16_autocast, validate_bf16_amp


@dataclass(frozen=True)
class GenerationProgress:
    """A streaming snapshot; final elapsed_seconds is the full request time.

    generated_tokens counts emitted ordinary token IDs and excludes special
    stop tokens. sampled_steps includes every sampled ID, including a stop ID.
    """

    text: str
    prompt_tokens: int
    context_tokens: int
    generated_tokens: int
    sampled_steps: int
    first_token_seconds: float | None
    elapsed_seconds: float
    complete: bool


def sample_top_p(logits: torch.Tensor, temperature: float = 1.0,
                 top_p: float = 0.8, generator=None) -> int:
    """Sample from the smallest descending-probability prefix reaching top_p."""
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("require temperature > 0 and 0 < top_p <= 1")
    probabilities = (logits.float() / temperature).softmax(-1)
    sorted_probs, indices = probabilities.sort(descending=True)
    # Include the token which crosses the threshold, and always the top token.
    keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
    filtered = sorted_probs * keep
    choice = torch.multinomial(filtered, 1, generator=generator)
    return indices[choice].item()


@torch.inference_mode()
def generate_stream(model: TransformerLM, tokenizer: BPE, prompt: str, *,
                    max_new_tokens: int = 512, temperature: float = 1.0,
                    top_p: float = 0.8, seed: int | None = None,
                    amp: bool = True) -> Iterator[GenerationProgress]:
    """Yield prompt plus completion after each token, then a final metric row.

    Stop on any sampled special ID. When context exceeds max_seq_len, use
    the most recent window and reset its RoPE positions to zero. No KV cache.
    first_token_seconds ends when the first ordinary token ID is sampled;
    elapsed_seconds on the final snapshot includes tokenization, generation,
    final device synchronization, and restoration of the model's train mode.
    """
    started = time.perf_counter()
    if max_new_tokens < 0 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("invalid generation length, temperature, or top_p")
    if model.vocab_size != max(tokenizer.vocab) + 1:
        raise ValueError("model and tokenizer vocabulary sizes do not match")
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must contain at least one token")
    device = next(model.parameters()).device
    validate_bf16_amp(device, amp)
    special_ids = set(tokenizer.special_tokens.values())
    output_ids = [i for i in ids if i not in special_ids]
    rng = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    context_limit = model.blocks[0].attention.max_seq_len
    context_ids = ids[-context_limit:]
    context = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
    was_training = model.training
    generated_tokens = 0
    sampled_steps = 0
    first_token_seconds = None
    decoder = getincrementaldecoder("utf-8")("replace")
    streamed_text = tokenizer.decode(output_ids)
    model.eval()
    try:
        for _ in range(max_new_tokens):
            with bf16_autocast(device, amp):
                logits = model.forward_last_logits(context)[0]
            token_id = sample_top_p(logits, temperature, top_p, rng)
            sampled_steps += 1
            if token_id in special_ids:
                break
            output_ids.append(token_id)
            token_bytes = tokenizer.vocab.get(token_id, "\ufffd".encode("utf-8"))
            streamed_text += decoder.decode(token_bytes, final=False)
            generated_tokens += 1
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - started
            next_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            context = torch.cat((context, next_token), dim=-1)[:, -context_limit:]
            yield GenerationProgress(
                text=streamed_text,
                prompt_tokens=len(ids),
                context_tokens=len(context_ids),
                generated_tokens=generated_tokens,
                sampled_steps=sampled_steps,
                first_token_seconds=first_token_seconds,
                elapsed_seconds=time.perf_counter() - started,
                complete=False,
            )
    finally:
        model.train(was_training)

    # Recompute the final text with the tokenizer's canonical UTF-8 replacement
    # behavior if generation ended between byte tokens.
    streamed_text = tokenizer.decode(output_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    yield GenerationProgress(
        text=streamed_text,
        prompt_tokens=len(ids),
        context_tokens=len(context_ids),
        generated_tokens=generated_tokens,
        sampled_steps=sampled_steps,
        first_token_seconds=first_token_seconds,
        elapsed_seconds=time.perf_counter() - started,
        complete=True,
    )


def generate_with_metrics(model: TransformerLM, tokenizer: BPE, prompt: str, *,
                          max_new_tokens: int = 512, temperature: float = 1.0,
                          top_p: float = 0.8, seed: int | None = None,
                          amp: bool = True) -> GenerationProgress:
    """Run a request and return its final token-level timing snapshot."""
    final = None
    for final in generate_stream(model, tokenizer, prompt,
                                 max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_p=top_p,
                                 seed=seed, amp=amp):
        pass
    if final is None:  # The stream always emits a final snapshot.
        raise RuntimeError("generation ended without a final metrics snapshot")
    return final


def generate(model: TransformerLM, tokenizer: BPE, prompt: str, *,
             max_new_tokens: int = 512, temperature: float = 1.0,
             top_p: float = 0.8, seed: int | None = None, amp: bool = True) -> str:
    """Return prompt plus completion, excluding every special token."""
    return generate_with_metrics(model, tokenizer, prompt,
                                 max_new_tokens=max_new_tokens,
                                 temperature=temperature, top_p=top_p,
                                 seed=seed, amp=amp).text


def load_model(checkpoint_path, tokenizer_path, device="auto", *,
               attention_backend="sdpa"):
    """Load a model and validate tokenizer identity.

    Inference defaults to SDPA; pass ``attention_backend="block"`` to use the
    educational tiled implementation.
    """
    from data_metadata import file_sha256
    target = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["extra"]["config"]
    tokenizer = BPE.from_file(tokenizer_path)
    tokenizer_id = "sha256:" + file_sha256(tokenizer_path)
    metadata = checkpoint["extra"].get("data_metadata")
    if not isinstance(metadata, dict) or not all(role in metadata for role in ("train", "val")):
        raise ValueError("checkpoint is missing train/val tokenizer identity metadata")
    for role in ("train", "val"):
        values = metadata[role]
        if not isinstance(values, dict) or values.get("tokenizer_id") != tokenizer_id:
            raise ValueError(f"{role} checkpoint tokenizer identity does not match")
        if values.get("vocab_size") != config["vocab_size"]:
            raise ValueError(f"{role} checkpoint vocabulary metadata does not match")
    if config["vocab_size"] != max(tokenizer.vocab) + 1:
        raise ValueError("checkpoint and tokenizer vocabulary sizes do not match")
    keys = ("vocab_size", "d_model", "num_layers", "num_heads", "d_ffn", "max_seq_len")
    model = TransformerLM(**{key: config[key] for key in keys}, device=target,
                          q_block_size=config.get("q_block_size", 64),
                          kv_block_size=config.get("kv_block_size", 64),
                          attention_backend=attention_backend)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention-backend", choices=("sdpa", "block"),
                        default="sdpa",
                        help="SDPA for faster inference; block keeps the teaching implementation")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    model, tokenizer = load_model(args.checkpoint, args.tokenizer_path, args.device,
                                  attention_backend=args.attention_backend)
    print(generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_p=args.top_p, seed=args.seed, amp=args.amp))


if __name__ == "__main__":
    main()
