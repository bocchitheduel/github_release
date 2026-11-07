"""Muon optimizer, Newton--Schulz orthogonalization, and parameter grouping.

The Newton--Schulz routine only computes an approximate polar factor. Shape
scaling is deliberately kept in :func:`compute_muon_scale` and applied by the
optimizer after orthogonalization.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from adamW import AdamW
from ffn_module import SwiGLU
from linear_module import Linear


DEFAULT_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
SCALING_MODES = ("rms_to_rms", "original", "rms_alignment", "match_rms_adamw")


def compute_muon_scale(fan_out: int, fan_in: int, *,
                       mode: str = "rms_to_rms") -> float:
    """Return the shape scale for a Muon update, independent of NS math.

    ``rms_to_rms`` is the operator-norm shape rule
    ``sqrt(fan_out / fan_in)``. ``rms_alignment`` uses
    ``0.2 * sqrt(max(fan_out, fan_in))`` to match update RMS to AdamW;
    ``match_rms_adamw`` remains an alias for compatibility. ``original``
    retains the original Muon shape rule. ``fan_out`` and ``fan_in`` refer
    to the logical linear map, regardless of whether its parameter is stored
    transposed.
    """
    if (not isinstance(fan_out, int) or isinstance(fan_out, bool) or fan_out <= 0
            or not isinstance(fan_in, int) or isinstance(fan_in, bool) or fan_in <= 0):
        raise ValueError("fan_out and fan_in must be positive integers")
    if mode == "rms_to_rms":
        return math.sqrt(fan_out / fan_in)
    if mode == "original":
        return math.sqrt(max(1.0, fan_out / fan_in))
    if mode in ("rms_alignment", "match_rms_adamw"):
        return 0.2 * math.sqrt(max(fan_out, fan_in))
    raise ValueError(f"unknown Muon scaling mode {mode!r}; expected one of {SCALING_MODES}")


def rms_to_rms_operator_norm(matrix: torch.Tensor) -> torch.Tensor:
    """Compute the exact RMS-to-RMS operator norm for a 2-D matrix."""
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("matrix must be non-empty and two-dimensional")
    m, n = matrix.shape
    work = matrix.to(torch.float64 if matrix.dtype == torch.float64 else torch.float32)
    return math.sqrt(n / m) * torch.linalg.matrix_norm(work, ord=2)


def newton_schulz_orthogonalize(matrix: torch.Tensor, ns_steps: int = 5, *,
                                 coefficients: tuple[float, float, float] =
                                 DEFAULT_NS_COEFFICIENTS,
                                 eps: float = 1e-7,
                                 validate_finite: bool = True) -> torch.Tensor:
    """Approximate the polar factor of a matrix using a quintic NS iteration.

    Low-precision inputs are promoted to float32; float64 inputs remain in
    float64. Tall matrices are transposed while iterating, so the Gram matrix
    uses the smaller dimension. The returned tensor is not shape-scaled.
    Zero inputs return zero. Non-finite inputs or results raise an error.
    """
    if matrix.ndim != 2 or min(matrix.shape) <= 0:
        raise ValueError("matrix must be non-empty and two-dimensional")
    if not matrix.is_floating_point():
        raise TypeError("matrix must have a floating-point dtype")
    if not isinstance(ns_steps, int) or isinstance(ns_steps, bool) or ns_steps < 0:
        raise ValueError("ns_steps must be a nonnegative integer")
    if len(coefficients) != 3 or any(not math.isfinite(float(value)) for value in coefficients):
        raise ValueError("coefficients must contain three finite values")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if validate_finite and not torch.isfinite(matrix).all():
        raise ValueError("Newton-Schulz input contains non-finite values")

    dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    x = matrix.to(dtype)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.transpose(0, 1)
    # Scale first so the Frobenius norm computation cannot overflow on large
    # finite values. Clamping also makes the zero matrix return zero without a
    # device-to-host synchronization.
    magnitude = x.abs().amax()
    safe_magnitude = magnitude.clamp_min(torch.finfo(dtype).tiny)
    x = x / safe_magnitude
    norm = torch.linalg.vector_norm(x)
    x = x / (norm + eps / safe_magnitude)
    a, b, c = (float(value) for value in coefficients)
    for _ in range(ns_steps):
        gram = x @ x.transpose(0, 1)
        gram_squared = gram @ gram
        polynomial = b * gram + c * gram_squared
        x = a * x + polynomial @ x
    if validate_finite and not torch.isfinite(x).all():
        raise FloatingPointError("Newton-Schulz produced non-finite values")
    return x.transpose(0, 1) if transposed else x


class Muon(Optimizer):
    """Momentum plus Newton--Schulz updates for two-dimensional weights."""

    def __init__(self, params: Iterable, lr: float = 1e-3, momentum: float = 0.95,
                 nesterov: bool = False, weight_decay: float = 0.01,
                 ns_steps: int = 5,
                 ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
                 eps: float = 1e-7, scaling_mode: str = "rms_to_rms"):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_steps=ns_steps,
                        ns_coefficients=tuple(ns_coefficients), eps=eps,
                        scaling_mode=scaling_mode)
        self._validate_group(defaults)
        super().__init__(params, defaults)

    @staticmethod
    def _validate_group(group):
        if not math.isfinite(group["lr"]) or group["lr"] < 0:
            raise ValueError("lr must be finite and nonnegative")
        if not math.isfinite(group["momentum"]) or not 0 <= group["momentum"] < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not math.isfinite(group["weight_decay"]) or group["weight_decay"] < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not math.isfinite(group["eps"]) or group["eps"] <= 0:
            raise ValueError("eps must be finite and positive")
        if not isinstance(group["ns_steps"], int) or group["ns_steps"] < 0:
            raise ValueError("ns_steps must be a nonnegative integer")
        coefficients = group["ns_coefficients"]
        if len(coefficients) != 3 or any(not math.isfinite(float(v)) for v in coefficients):
            raise ValueError("ns_coefficients must contain three finite values")
        if group["scaling_mode"] not in SCALING_MODES:
            raise ValueError(f"scaling_mode must be one of {SCALING_MODES}")
        if not isinstance(group["nesterov"], bool):
            raise ValueError("nesterov must be a bool")
        for parameter in group.get("params", ()):
            if parameter.ndim != 2:
                raise ValueError("Muon parameters must be two-dimensional matrices")

    def add_param_group(self, param_group):
        self._validate_group({**getattr(self, "defaults", {}), **param_group})
        super().add_param_group(param_group)

    def load_state_dict(self, state_dict):
        """Restore momentum in fp32 for low-precision model parameters."""
        saved_buffers = {}

        def preserve_buffers(optimizer, incoming):
            copied = dict(incoming)
            copied["state"] = {}
            saved_buffers["groups"] = incoming["param_groups"]
            for key, value in incoming["state"].items():
                copied["state"][key] = dict(value)
                if "momentum_buffer" in copied["state"][key]:
                    saved_buffers[key] = copied["state"][key].pop("momentum_buffer")
            return copied

        def restore_buffers(optimizer):
            for saved_group, group in zip(saved_buffers["groups"], optimizer.param_groups):
                for key, parameter in zip(saved_group["params"], group["params"]):
                    value = saved_buffers.get(key)
                    if value is not None:
                        dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
                        optimizer.state[parameter]["momentum_buffer"] = value.to(
                            device=parameter.device, dtype=dtype).clone()

        pre_hook = self.register_load_state_dict_pre_hook(preserve_buffers)
        post_hook = self.register_load_state_dict_post_hook(restore_buffers, prepend=True)
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
        active = []
        gradients_by_device = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if gradient.ndim != 2 or gradient.shape != parameter.shape:
                    raise ValueError("Muon gradient shape must match its matrix parameter")
                active.append((group, parameter, gradient))
                gradients_by_device.setdefault(gradient.device, []).append(gradient)
        # One synchronization per device, rather than one per matrix. The
        # training loop also rejects non-finite global gradients before step.
        for gradients in gradients_by_device.values():
            finite = torch.stack([torch.isfinite(gradient).all()
                                  for gradient in gradients]).all()
            if not finite:
                raise ValueError("Muon gradient contains non-finite values")

        for group, parameter, gradient in active:
            state = self.state[parameter]
            state_dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
            if not state:
                state["step"] = 0
                state["momentum_buffer"] = torch.zeros_like(parameter, dtype=state_dtype)
            state["step"] += 1
            buffer = state["momentum_buffer"]
            grad = gradient.to(dtype=state_dtype)
            buffer.mul_(group["momentum"]).add_(grad)
            direction = (grad.add(buffer, alpha=group["momentum"])
                         if group["nesterov"] else buffer)
            orthogonal = newton_schulz_orthogonalize(
                direction, group["ns_steps"],
                coefficients=group["ns_coefficients"], eps=group["eps"],
                validate_finite=False)
            # Scale is separate from Newton--Schulz: it depends only on
            # the logical Linear dimensions, not the iteration itself.
            scale = compute_muon_scale(
                parameter.shape[0], parameter.shape[1], mode=group["scaling_mode"])
            if group["weight_decay"]:
                # Keep decay outside the shape multiplier: the resulting
                # update is -lr * (scale * orthogonal + weight_decay * W).
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
            parameter.add_(orthogonal.to(parameter.dtype), alpha=-group["lr"] * scale)
        return loss


class OptimizerBundle:
    """Coordinate a Muon/AdamW pair through one small optimizer-like API."""

    def __init__(self, optimizers: Iterable[Optimizer]):
        self.optimizers = list(optimizers)
        if not self.optimizers:
            raise ValueError("OptimizerBundle requires at least one optimizer")
        self._refresh_groups()

    def _refresh_groups(self):
        self.param_groups = [group for optimizer in self.optimizers
                             for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("OptimizerBundle does not support closures")
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return {
            "format": "optimizer_bundle_v1",
            "optimizers": [
                {"class": type(optimizer).__name__, "state": optimizer.state_dict()}
                for optimizer in self.optimizers
            ],
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("format") != "optimizer_bundle_v1":
            raise ValueError("checkpoint does not contain an optimizer bundle")
        saved = state_dict.get("optimizers", [])
        if len(saved) != len(self.optimizers):
            raise ValueError("optimizer bundle component count does not match checkpoint")
        for optimizer, component in zip(self.optimizers, saved):
            if component.get("class") != type(optimizer).__name__:
                raise ValueError("optimizer bundle component type does not match checkpoint")
        for optimizer, component in zip(self.optimizers, saved):
            optimizer.load_state_dict(component["state"])
        self._refresh_groups()


_EXCLUDED_NAME_COMPONENTS = {
    "embedding", "token_embedding", "position_embedding", "lm_head",
    "output", "classifier", "vocab_projection",
}


def partition_muon_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (Muon, AdamW) parameter lists with conservative name handling.

    Only known Linear projections and the three explicit SwiGLU projections
    are candidates. Embedding/output/norm/bias/1-D/unknown parameters stay on
    AdamW. If a tied parameter has any excluded alias, AdamW owns it.
    """
    candidates: set[int] = set()
    excluded_aliases: set[int] = set()
    for module_name, module in model.named_modules(remove_duplicate=False):
        components = set(module_name.lower().split(".")) if module_name else set()
        direct = list(module.named_parameters(recurse=False))
        for local_name, parameter in direct:
            if not parameter.requires_grad:
                continue
            if components & _EXCLUDED_NAME_COMPONENTS:
                excluded_aliases.add(id(parameter))
                continue
            is_linear_projection = (
                isinstance(module, (Linear, nn.Linear))
                and local_name in ("W", "weight")
                and parameter.ndim == 2
            )
            is_swiglu_projection = (
                isinstance(module, SwiGLU)
                and local_name in ("W_gate", "W_up", "W_down")
                and parameter.ndim == 2
            )
            if is_linear_projection or is_swiglu_projection:
                candidates.add(id(parameter))

    muon_ids = candidates - excluded_aliases
    seen = set()
    muon_params, adamw_params = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (muon_params if id(parameter) in muon_ids else adamw_params).append(parameter)
    return muon_params, adamw_params


def create_optimizers(model: nn.Module, config) -> Optimizer | OptimizerBundle:
    """Build AdamW or Muon+AdamW from a TrainConfig or mapping."""
    def get(name, default=None):
        return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)

    kind = get("optimizer", "adamw")
    lr = get("max_lr", 1e-3)
    weight_decay = get("weight_decay", 0.01)
    if kind == "adamw":
        adamw_lr = get("adamw_lr", None)
        adamw_weight_decay = get("adamw_weight_decay", None)
        optimizer = AdamW(model.parameters(),
                          lr=lr if adamw_lr is None else adamw_lr,
                          betas=(get("beta1", 0.9), get("beta2", 0.999)),
                          weight_decay=(weight_decay if adamw_weight_decay is None
                                        else adamw_weight_decay))
        return optimizer
    if kind not in ("muon_rms_to_rms", "muon_original"):
        raise ValueError("optimizer must be adamw, muon_rms_to_rms, or muon_original")
    muon_params, adamw_params = partition_muon_parameters(model)
    optimizers = []
    if muon_params:
        scaling_mode = ("original" if kind == "muon_original"
                        else get("muon_scaling_mode", "rms_to_rms"))
        muon_lr = get("muon_lr", None)
        muon_weight_decay = get("muon_weight_decay", None)
        optimizers.append(Muon(
            muon_params, lr=lr if muon_lr is None else muon_lr,
            momentum=get("muon_momentum", 0.95),
            nesterov=get("muon_nesterov", False),
            weight_decay=(weight_decay if muon_weight_decay is None else muon_weight_decay),
            ns_steps=get("muon_ns_steps", 5),
            ns_coefficients=get("muon_ns_coefficients", DEFAULT_NS_COEFFICIENTS),
            eps=get("muon_eps", 1e-7), scaling_mode=scaling_mode))
    if adamw_params:
        adamw_lr = get("adamw_lr", None)
        adamw_weight_decay = get("adamw_weight_decay", None)
        optimizers.append(AdamW(
            adamw_params, lr=lr if adamw_lr is None else adamw_lr,
            betas=(get("beta1", 0.9), get("beta2", 0.999)),
            weight_decay=(weight_decay if adamw_weight_decay is None
                          else adamw_weight_decay)))
    if not optimizers:
        raise ValueError("model has no trainable parameters")
    return optimizers[0] if len(optimizers) == 1 else OptimizerBundle(optimizers)
