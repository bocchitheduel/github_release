"""Train batch 128 x sequence 256 for 10,000 updates with TF32 + BF16 AMP.

Run with this project's .venv/bin/python in WSL. No training starts on import.
Use --dry-run to inspect the fixed configuration without training.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import asdict, fields, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import uuid
from datetime import datetime
from pathlib import Path

def make_run_dir(base: Path, name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    uid = uuid.uuid4().hex[:6]
    return base / f"{name}_{ts}_{uid}"

REPO = Path(__file__).resolve().parent
METRIC_FIELDS = (
    "timestamp", "session_id", "step", "tokens_seen", "lr", "window_steps", "train_loss",
    "train_perplexity", "train_loss_last_batch", "val_loss", "val_perplexity",
    "grad_norm", "train_seconds", "tokens_per_second",
)


def perplexity(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


class Tee:
    """Mirror Python output to a flushed log while retaining terminal behavior."""

    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        result = self.terminal.write(text)
        self.log.write(text)
        self.log.flush()
        return result

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class MetricsWriter:
    """One CSV row per log interval; a resume appends with an explicit marker."""

    def __init__(self, stream, session_id: str, total_steps: int = 10_000):
        self.stream = stream
        self.session_id = session_id
        self.total_steps = total_steps
        self.writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        if stream.tell() == 0:
            self.writer.writeheader()
            stream.flush()

    def __call__(self, metrics: dict) -> None:
        row = dict(metrics)
        row["timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
        row["session_id"] = self.session_id
        row["train_perplexity"] = perplexity(row["train_loss"])
        row["val_perplexity"] = perplexity(row["val_loss"])
        self.writer.writerow(row)
        self.stream.flush()
        validation = ""
        if row["val_loss"] is not None:
            validation = (f" val_loss={row['val_loss']:.6f}"
                          f" val_ppl={row['val_perplexity']:.4f}")
        print(f"step={row['step']:5d}/{self.total_steps} tokens={row['tokens_seen']:,}"
              f" lr={row['lr']:.7g} train_loss={row['train_loss']:.6f}"
              f" train_ppl={row['train_perplexity']:.4f}{validation}"
              f" tok/s={row['tokens_per_second']:,.0f}"
              f" window_steps={row['window_steps']}", flush=True)


def build_config(output: Path, resume: Path | None, *, optimizer: str = "adamw",
                 muon_scaling_mode: str = "rms_to_rms",
                 gradient_accumulation_steps: int = 1,
                 config_path: Path | None = None):
    from train_loop import TrainConfig

    config = TrainConfig(
        train_path=str(REPO / "train.bin"),
        val_path=str(REPO / "val.bin"),
        vocab_size=8983,
        output_dir=str(output / "checkpoints"),
        resume=str(resume) if resume else None,
        device="cuda:0",
        seed=42,
        batch_size=128,
        sequence_length=256,
        gradient_accumulation_steps=gradient_accumulation_steps,
        d_model=512,
        d_ffn=1344,
        num_layers=4,
        num_heads=16,
        max_seq_len=512,
        total_steps=10_000,
        warmup_steps=500,
        max_lr=1e-3,
        min_lr=1e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        optimizer=optimizer,
        muon_scaling_mode=muon_scaling_mode,
        grad_clip=1.0,
        log_interval=100,
        eval_interval=100,
        eval_batches=20,
        save_interval=1000,
        compile=True,
        compile_backend="inductor",
        compile_mode="default",
        q_block_size=64,
        kv_block_size=64,
        amp=True,
    )
    if config_path is None:
        return config
    try:
        import yaml
    except ImportError as error:
        raise ValueError("loading --config requires PyYAML in the project environment") from error
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("training YAML must contain a mapping")
    aliases = {
        "total_training_steps": "total_steps",
        "gradient_clip_norm": "grad_clip",
    }
    values = {aliases.get(key, key): value for key, value in raw.items()}
    values.pop("name", None)
    values.pop("search_validation_loss", None)
    values.pop("search_training_steps", None)
    adjust_lr_fn = values.pop("adjust_lr_fn", None)
    if adjust_lr_fn not in (None, "spectral_unclamped"):
        raise ValueError("this training script supports only RMS-to-RMS spectral_unclamped Muon scaling")
    if adjust_lr_fn == "spectral_unclamped":
        values["muon_scaling_mode"] = "rms_to_rms"
    betas = values.pop("adamw_betas", None)
    if betas is not None:
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError("adamw_betas must contain two values")
        values["beta1"], values["beta2"] = betas
    if isinstance(values.get("amp"), str):
        if values["amp"].lower() != "bfloat16":
            raise ValueError("only bfloat16 AMP is supported by this configuration")
        values["amp"] = True
    if "muon_ns_coefficients" in values:
        values["muon_ns_coefficients"] = tuple(values["muon_ns_coefficients"])
    definitions = {field.name: field for field in fields(TrainConfig)}
    float_fields = {"muon_lr", "muon_weight_decay", "adamw_lr", "adamw_weight_decay"}
    for key, value in list(values.items()):
        default = definitions.get(key).default if key in definitions else None
        if isinstance(value, str) and (isinstance(default, float) or key in float_fields):
            values[key] = float(value)
    valid_fields = set(definitions)
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        raise ValueError(f"unsupported training YAML fields: {', '.join(unknown)}")
    return replace(config, **values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="new WSL output directory; defaults to runs/b128_s256_10k_<time>")
    parser.add_argument("--config", type=Path,
                        help="load an optional YAML training configuration")
    parser.add_argument("--resume", type=Path, metavar="CHECKPOINT",
                        help="resume a checkpoint created with this 10,000-step schedule")
    parser.add_argument("--optimizer", choices=("adamw", "muon_rms_to_rms", "muon_original"),
                        default="adamw", help="optimizer path; Muon uses AdamW for embeddings/norms/output")
    parser.add_argument("--muon-scaling-mode", choices=("rms_to_rms", "original", "rms_alignment", "match_rms_adamw"),
                        default="rms_to_rms", help="shape scaling for muon_rms_to_rms experiments")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="microbatches per optimizer update")
    parser.add_argument("--dry-run", action="store_true",
                        help="print configuration; do not train or create run files")
    args = parser.parse_args()
    resume = args.resume.expanduser().resolve() if args.resume else None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.output_dir:
        output = args.output_dir.expanduser().resolve()
    elif resume:
        output = resume.parent.parent if resume.parent.name == "checkpoints" else resume.parent
    else:
        output = REPO / "runs" / f"b128_s256_10k_{stamp}"
    if resume is not None and not resume.is_file():
        parser.error(f"checkpoint does not exist: {resume}")
    if not args.dry_run and not resume and output.exists():
        parser.error("output directory already exists; choose a new directory or use --resume")
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)

    # Configure compiler/BLAS workers before importing PyTorch. These defaults
    # match the successful compiled benchmark on this WSL machine.
    environment_defaults = {
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TORCHINDUCTOR_CACHE_DIR": str(output / "cache" / "inductor"),
        "TRITON_CACHE_DIR": str(output / "cache" / "triton"),
    }
    environment = {name: os.environ.get(name, value)
                   for name, value in environment_defaults.items()}
    for name, value in environment.items():
        # Importing Inductor can create its cache directory. A dry-run reports
        # the intended paths without creating a production run directory.
        if not args.dry_run or name not in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            os.environ.setdefault(name, value)

    import torch
    import torch._dynamo
    from amp_utils import validate_bf16_amp
    from train_loop import train
    from train_utils import cosine_learning_rate

    config = build_config(output, resume, optimizer=args.optimizer,
                          muon_scaling_mode=args.muon_scaling_mode,
                          gradient_accumulation_steps=args.gradient_accumulation_steps,
                          config_path=args.config.expanduser().resolve() if args.config else None)
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this TF32 + BF16 AMP training configuration")
    torch.cuda.set_device(0)
    validate_bf16_amp(torch.device(config.device), config.amp)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False

    manifest = {
        "dry_run": args.dry_run,
        "started_at": datetime.now().astimezone().isoformat(),
        "config": asdict(config),
        "total_tokens": (config.total_steps * config.batch_size * config.sequence_length
                         * config.gradient_accumulation_steps),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "amp_dtype": "bfloat16",
        "environment": environment,
        "lr_by_zero_based_iteration": {
            str(step): ((config.muon_lr if config.muon_lr is not None else config.max_lr)
                        * (min((step + 1) / config.warmup_steps, 1.0)
                           if config.scheduler == "warmup_constant" and config.warmup_steps
                           else (1.0 if config.scheduler == "warmup_constant" else
                                 (cosine_learning_rate(step, config.max_lr, config.min_lr,
                                                       config.lr_schedule_steps or config.total_steps,
                                                       config.warmup_steps)
                                  / config.max_lr if config.max_lr else 0.0))))
            for step in (0, 9, 10, 100, 499, 500, 1000, 5250, 9999, 10000)
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(REPO.glob("*.py"))
        },
        "metrics_notes": {
            "train_loss": "mean over window_steps since the previous log or resume",
            "perplexity": "exp(mean cross-entropy), using natural-log loss",
            "val_loss": "mean over 20 validation batches with reproducible sampling",
            "tokens_per_second": "training only; first window includes lazy compilation",
            "resume": "CSV appends; repeated steps can appear when replaying after a checkpoint",
        },
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    for path in (config.train_path, config.val_path):
        if not Path(path).is_file():
            parser.error(f"token file does not exist: {path}")
    # output.mkdir(parents=True, exist_ok=resume is not None)
    manifest_name = f"session_{stamp}.json" if resume else "run_config.json"
    (output / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (output / "training.log").open("a", encoding="utf-8", buffering=1) as log, \
            (output / "metrics.csv").open("a", encoding="utf-8", newline="") as csv_stream, \
            redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        print(f"Run directory: {output}", flush=True)
        total_tokens = (config.total_steps * config.batch_size * config.sequence_length
                        * config.gradient_accumulation_steps)
        print(f"microbatch={config.batch_size} seq={config.sequence_length} "
              f"accumulation={config.gradient_accumulation_steps} "
              f"updates={config.total_steps} tokens={total_tokens:,}; optimizer={config.optimizer}; "
              "Inductor + TF32 + BF16 AMP", flush=True)
        if config.scheduler == "warmup_constant":
            print(f"LR: {config.warmup_steps} warmup updates, then constant; "
                  f"Muon={config.muon_lr or config.max_lr:g}, "
                  f"AdamW={config.adamw_lr or config.max_lr:g}. "
                  "Log/evaluate every 100; save latest checkpoint every 1000.", flush=True)
        else:
            print(f"LR: {config.warmup_steps} warmup updates, then cosine decay; "
                  f"horizon={config.lr_schedule_steps or config.total_steps} updates, "
                  f"max={config.max_lr:g} min={config.min_lr:g}. "
                  "Log/evaluate every 100; save latest checkpoint every 1000.", flush=True)
        if resume:
            print(f"Resuming {resume}; CSV appends a new session.", flush=True)
        print("train_loss is the interval mean; first throughput includes compilation.",
              flush=True)
        try:
            result = train(config, on_metrics=MetricsWriter(csv_stream, stamp, config.total_steps))
        except BaseException:
            import traceback
            traceback.print_exc(file=log)
            raise
        (output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Completed {result['step']} updates. Checkpoint: {result['checkpoint']}",
              flush=True)


if __name__ == "__main__":
    main()
