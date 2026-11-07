"""Manual AdamW with decoupled weight decay."""
import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """Adam with bias correction and decoupled parameter decay.

    Supports parameter groups, closures, and optimizer state_dict checkpoints.
    Moments for float16/bfloat16 parameters are stored in float32.
    Parameters with no gradient are skipped, including weight decay.
    Initialize or load optimizer state after placing parameters on their final
    device/dtype; active parameter placement must remain unchanged during training.
    """

    def __init__(self, params: Iterable, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay))
        # Runtime scratch buffers are not optimizer state/checkpoint contents.
        self._update_buffers = {}

    def add_param_group(self, param_group):
        # Validate once before adding a new group (also used by __init__).
        self._validate_group({**self.defaults, **param_group})
        super().add_param_group(param_group)

    @staticmethod
    def _validate_group(group):
        if not math.isfinite(group["lr"]) or group["lr"] < 0:
            raise ValueError("lr must be finite and nonnegative")
        if not math.isfinite(group["eps"]) or group["eps"] <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if len(group["betas"]) != 2 or any(not 0 <= b < 1 for b in group["betas"]):
            raise ValueError("betas must contain two values in [0, 1)")

    def load_state_dict(self, state_dict):
        """Restore moments without first casting them to parameter precision.

        Keep the base optimizer's group validation and hook behavior, but
        remove moment tensors from its automatic dtype conversion. Restore
        them directly from saved values before other post-load hooks run.
        """
        saved_moments = {}

        def preserve_moments(optimizer, incoming):
            for group in incoming["param_groups"]:
                optimizer._validate_group(group)
            copied = dict(incoming)
            copied["state"] = {}
            for key, value in incoming["state"].items():
                copied["state"][key] = dict(value)
                saved_moments[key] = {}
                for name in ("exp_avg", "exp_avg_sq"):
                    if name in copied["state"][key]:
                        saved_moments[key][name] = copied["state"][key].pop(name)
            saved_moments["groups"] = incoming["param_groups"]
            return copied

        def restore_moments(optimizer):
            for saved_group, group in zip(saved_moments["groups"], optimizer.param_groups):
                for key, parameter in zip(saved_group["params"], group["params"]):
                    dtype = (torch.float32 if parameter.dtype in (torch.float16, torch.bfloat16)
                             else parameter.dtype)
                    for name, value in saved_moments.get(key, {}).items():
                        optimizer.state[parameter][name] = value.to(
                            device=parameter.device, dtype=dtype).clone()
            optimizer._update_buffers.clear()

        pre_hook = self.register_load_state_dict_pre_hook(preserve_moments)
        post_hook = self.register_load_state_dict_post_hook(restore_moments, prepend=True)
        try:
            super().load_state_dict(state_dict)
        finally:
            pre_hook.remove()
            post_hook.remove()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps = group["lr"], group["eps"]
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    if not parameter.is_floating_point():
                        raise TypeError("AdamW requires real floating-point parameters")
                    dtype = (torch.float32 if parameter.dtype in (torch.float16, torch.bfloat16)
                             else parameter.dtype)
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, dtype=dtype)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=dtype)
                state["step"] += 1
                step = state["step"]
                moment, variance = state["exp_avg"], state["exp_avg_sq"]
                buffers = self._update_buffers.get(parameter)
                if buffers is None:
                    # Decide once whether this parameter needs promoted math.
                    if parameter.dtype == moment.dtype:
                        buffers = (None, None)
                    else:
                        buffers = (torch.empty_like(moment), torch.empty_like(moment))
                    self._update_buffers[parameter] = buffers
                gradient_buffer, parameter_buffer = buffers
                if gradient_buffer is None:
                    gradient = parameter.grad
                    updated = parameter
                else:
                    # copy_ performs the necessary cast into reusable storage,
                    # without allocating converted tensors at every iteration.
                    gradient_buffer.copy_(parameter.grad)
                    parameter_buffer.copy_(parameter)
                    gradient = gradient_buffer
                    updated = parameter_buffer
                moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                # Fold first-moment bias correction into the scalar step size,
                # avoiding a full-sized temporary and elementwise division.
                step_size = lr / (1 - beta1**step)
                # Keep the stored variance intact; reuse one scratch tensor
                # for sqrt, second-moment correction, and epsilon addition.
                denominator = variance.sqrt()
                denominator.div_(math.sqrt(1 - beta2**step)).add_(eps)
                updated.mul_(1 - lr * group["weight_decay"])
                updated.addcdiv_(moment, denominator, value=-step_size)
                if updated is not parameter:
                    parameter.copy_(updated)
        return loss
