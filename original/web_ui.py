"""Gradio Web UI for interactive text generation with the trained BPE TransformerLM.

Reuses `generate_stream()` / `load_model()` from generate.py, so sampling,
token-level timing, special-token handling and checkpoint validation stay in
one place while each generated token is sent to the UI.

Usage:
    .venv/bin/python web_ui.py \
        --tokenizer-path /path/to/vocab.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import gradio as gr

# Reuse the tested generation utilities from generate.py
from generate import generate_stream, load_model

# ---------------------------------------------------------------------------
# Model cache — load once, reuse across every request
# ---------------------------------------------------------------------------
_MODEL = None
_TOKENIZER = None
_DEVICE = "cpu"
_CKPT = None
_TOK = None
_MODEL_KEY = None


def _resolve_device(device: str) -> str:
    if device == "auto":
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        index = target.index if target.index is not None else torch.cuda.current_device()
        return f"cuda:{index}"
    return str(target)


def load_once(checkpoint_path: str, tokenizer_path: str, device: str,
              attention_backend: str = "sdpa"):
    global _MODEL, _TOKENIZER, _DEVICE, _CKPT, _TOK, _MODEL_KEY
    ckpt = str(Path(checkpoint_path).expanduser().resolve())
    tok = str(Path(tokenizer_path).expanduser().resolve())
    resolved_device = _resolve_device(device)
    cache_key = (ckpt, tok, resolved_device, attention_backend)
    if _MODEL is not None and _MODEL_KEY == cache_key:
        return _MODEL, _TOKENIZER

    print(f"[load] checkpoint = {ckpt}")
    print(f"[load] tokenizer  = {tok}")
    print(f"[load] device     = {resolved_device}")
    print(f"[load] attention  = {attention_backend}")

    _MODEL, _TOKENIZER = load_model(
        ckpt, tok, device=resolved_device,
        attention_backend=attention_backend)
    _DEVICE = next(_MODEL.parameters()).device
    _CKPT, _TOK, _MODEL_KEY = ckpt, tok, cache_key
    print(f"[load] done, model on {_DEVICE}")
    return _MODEL, _TOKENIZER


# ---------------------------------------------------------------------------
# Generation callback
# ---------------------------------------------------------------------------
def run_generation(prompt: str,
                   max_new_tokens: int,
                   temperature: float,
                   top_p: float,
                   seed,
                   amp: bool):
    if _MODEL is None or _TOKENIZER is None:
        yield "", "❌ 模型尚未加载，请检查命令行参数。"
        return
    if not prompt or not prompt.strip():
        yield "", "❌ Prompt 不能为空。"
        return

    seed_val = None
    try:
        if seed is not None and int(seed) >= 0:
            seed_val = int(seed)
    except (TypeError, ValueError):
        seed_val = None

    try:
        yield "", "生成中 · 正在准备 prompt 并计算首 token…"
        updates = generate_stream(
            _MODEL,
            _TOKENIZER,
            prompt,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            seed=seed_val,
            amp=bool(amp),
        )
        for update in updates:
            if update.first_token_seconds is None:
                ttft = "—"
            else:
                ttft = f"{1000 * update.first_token_seconds:.0f} ms"
            elapsed = update.elapsed_seconds
            speed = (update.generated_tokens / elapsed) if elapsed > 0 else 0.0
            phase = "完成" if update.complete else "生成中"
            elapsed_label = "请求总耗时" if update.complete else "已用时"
            info = (
                f"{phase} · 输出 {update.generated_tokens} token · "
                f"首 token {ttft} · {elapsed_label} {elapsed:.2f}s "
                f"（{speed:.1f} token/s） · 输入 {update.prompt_tokens} token "
                f"（初始上下文 {update.context_tokens}） · 设备 `{_DEVICE}` · "
                f"注意力 `{_MODEL.blocks[0].attention.attention_backend}` · "
                f"seed: `{seed_val if seed_val is not None else 'random'}`"
            )
            yield update.text, info
    except Exception as exc:  # noqa: BLE001 — surface any error into the UI
        yield "", f"❌ 生成失败: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui(default_prompt: str) -> gr.Blocks:
    with gr.Blocks(
        title="CS336 · 10k-step TransformerLM",
    ) as demo:
        gr.Markdown(
            "# CS336 · b128_s256_10k · 交互式文本生成\n"
            "Temperature + top-p 采样 · 直接复用 `generate.py` 中的推理逻辑。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=default_prompt,
                    lines=5,
                    placeholder="在这里输入提示词…",
                    autofocus=True,
                )
                with gr.Row():
                    btn = gr.Button("生成", variant="primary", scale=2)
                    clear_btn = gr.ClearButton(scale=1)

            with gr.Column(scale=2):
                max_new_tokens = gr.Slider(
                    1, 1024, value=200, step=1, label="Max new tokens"
                )
                temperature = gr.Slider(
                    0.1, 2.0, value=0.8, step=0.05, label="Temperature"
                )
                top_p = gr.Slider(
                    0.05, 1.0, value=0.9, step=0.01, label="Top-p (nucleus)"
                )
                seed = gr.Number(value=42, precision=0,
                                 label="Seed（填 -1 表示随机）")
                amp = gr.Checkbox(value=True, label="BF16 autocast (AMP)")

                output = gr.Textbox(
                    label="生成结果",
                    lines=20,
                    autoscroll=False,
                )
        info = gr.Markdown("")

        clear_btn.add([output, info])

        inputs = [prompt, max_new_tokens, temperature, top_p, seed, amp]
        outputs = [output, info]
        btn.click(run_generation, inputs=inputs, outputs=outputs)
        prompt.submit(run_generation, inputs=inputs, outputs=outputs)

        gr.Examples(
            examples=[
                ["Once upon a time"],
                ["The meaning of life is"],
                ["In a world where machines could think, "],
                ["Dear diary, today I"],
            ],
            inputs=[prompt],
        )
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="./runs/"
                "b128_s256_10k_20261015_171649_473135/checkpoints/latest.pt",
        help="训练好的 checkpoint（.pt）",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="训练时使用的 BPE 分词器数据文件（例如 vocab.json）",
    )
    parser.add_argument("--device", default="auto",
                        help="auto / cuda / cuda:0 / cpu")
    parser.add_argument("--attention-backend", choices=("sdpa", "block"),
                        default="sdpa",
                        help="SDPA for faster inference; block keeps the teaching implementation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="生成 gradio.live 公网链接")
    parser.add_argument("--default-prompt", default="Once upon a time")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).expanduser().resolve()
    tok = Path(args.tokenizer_path).expanduser().resolve()
    if not ckpt.is_file():
        parser.error(f"checkpoint 不存在: {ckpt}")
    if not tok.is_file():
        parser.error(f"tokenizer 数据文件不存在: {tok}")

    load_once(str(ckpt), str(tok), args.device, args.attention_backend)

    demo = build_ui(args.default_prompt)
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=False,
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px !important;}",
    )


if __name__ == "__main__":
    main()
