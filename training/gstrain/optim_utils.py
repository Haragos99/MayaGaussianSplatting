"""Adam state surgery for adding and removing Gaussians.

Params and their `exp_avg` / `exp_avg_sq` moments must be grown and masked in
lockstep, otherwise the optimizer silently keeps stale momentum for rows that no
longer exist. `step` is left alone so the bias correction stays continuous.
"""

from __future__ import annotations

import torch
from torch import nn


def _group(optimizer, name):
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return group
    raise KeyError(f"no optimizer param group named {name!r}")


def _rebind(optimizer, group, tensor: torch.Tensor, state) -> nn.Parameter:
    old = group["params"][0]
    optimizer.state.pop(old, None)
    param = nn.Parameter(tensor.requires_grad_(True))
    group["params"][0] = param
    if state is not None:
        optimizer.state[param] = state
    return param


def replace_tensor_to_optimizer(optimizer, tensor: torch.Tensor, name: str) -> nn.Parameter:
    """Swap one parameter wholesale and drop its momentum (used by opacity reset)."""
    group = _group(optimizer, name)
    state = optimizer.state.get(group["params"][0])
    if state is not None:
        state = dict(state)
        state["exp_avg"] = torch.zeros_like(tensor)
        state["exp_avg_sq"] = torch.zeros_like(tensor)
    return _rebind(optimizer, group, tensor, state)


def cat_tensors_to_optimizer(optimizer, tensors: dict[str, torch.Tensor]) -> dict[str, nn.Parameter]:
    """Append rows to several parameters at once; new rows start with zero momentum."""
    out: dict[str, nn.Parameter] = {}
    for name, extension in tensors.items():
        group = _group(optimizer, name)
        old = group["params"][0]
        state = optimizer.state.get(old)
        if state is not None:
            state = dict(state)
            zeros = torch.zeros_like(extension)
            state["exp_avg"] = torch.cat([state["exp_avg"], zeros], dim=0)
            state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.zeros_like(extension)], dim=0)
        merged = torch.cat([old.data, extension], dim=0)
        out[name] = _rebind(optimizer, group, merged, state)
    return out


def prune_optimizer(optimizer, keep: torch.Tensor) -> dict[str, nn.Parameter]:
    """Keep only the rows selected by the boolean mask, momentum included."""
    out: dict[str, nn.Parameter] = {}
    for group in optimizer.param_groups:
        name = group.get("name")
        if name is None:
            continue
        old = group["params"][0]
        state = optimizer.state.get(old)
        if state is not None:
            state = dict(state)
            state["exp_avg"] = state["exp_avg"][keep]
            state["exp_avg_sq"] = state["exp_avg_sq"][keep]
        out[name] = _rebind(optimizer, group, old.data[keep], state)
    return out
