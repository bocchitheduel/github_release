"""BF16 automatic mixed precision policy shared by training and generation."""
import torch


def validate_bf16_amp(device: torch.device, enabled: bool) -> None:
    if not enabled:
        return
    if device.type not in ("cpu", "cuda"):
        raise ValueError("BF16 AMP supports CPU/CUDA here; disable AMP for other devices")
    if device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("CUDA device does not support BF16; use --no-amp")


def bf16_autocast(device: torch.device, enabled: bool = True):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)
