"""Checkpoint load/save helpers for single-task and multi-task PoEFuse runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_poe_fuse_checkpoint(
    model: nn.Module,
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
    strict: bool = False,
) -> dict[str, Any]:
    """Load a single-task checkpoint (``train.py`` format) into ``model``."""
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=strict)
        return state
    if isinstance(state, dict):
        model.load_state_dict(state, strict=strict)
        return {"model": state}
    raise ValueError(f"unrecognised checkpoint at {path}")


def load_multitask_checkpoint(
    bundles: list[Any],
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load ``{trunk, heads}`` checkpoint produced by ``train_multitask.py``.

    ``bundles`` is a list of objects with ``.key`` and ``.model`` (codec + head).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "trunk" not in state:
        raise ValueError(f"expected multitask checkpoint with 'trunk' key at {path}")
    bundles[0].model.codec.load_state_dict(state["trunk"], strict=False)
    heads = state.get("heads") or {}
    adapters = state.get("adapters") or {}
    for b in bundles:
        if b.key in heads:
            b.model.head.load_state_dict(heads[b.key], strict=False)
        if b.key in adapters and b.model.adapter is not None:
            b.model.adapter.load_state_dict(adapters[b.key], strict=False)
    return state


def save_multitask_checkpoint(
    bundles: list[Any],
    path: Path | str,
    *,
    epoch: int,
    scores: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adapters = {b.key: b.model.adapter.state_dict() for b in bundles if b.model.adapter is not None}
    payload: dict[str, Any] = {
        "trunk": bundles[0].model.codec.state_dict(),
        "heads": {b.key: b.model.head.state_dict() for b in bundles},
        "epoch": epoch,
        "scores": scores,
    }
    if adapters:
        payload["adapters"] = adapters
    if extra:
        payload.update(extra)
    torch.save(payload, path)


__all__ = [
    "load_multitask_checkpoint",
    "load_poe_fuse_checkpoint",
    "save_multitask_checkpoint",
]
