"""Sequence-mixer zoo for the mixer ablation.

``build_mixer`` returns a drop-in replacement for :class:`Mamba3Stack`:
``(B, L, d_model) -> (B, L, d_model)``.  All kinds are parameter-matched to
the default 4-layer Mamba-3 stack (~25-27M at d_model=1024) via their default
depths (see ``MIXER_DEFAULT_LAYERS``):

* ``mamba3``       -- the existing unidirectional Mamba-3 stack (4 layers).
* ``mamba3_bidir`` -- bidirectional: per layer a forward and a backward
  Mamba-3 block with *direction-specific* weights, outputs averaged
  (2 layers x 2 directions = param-matched to 4 unidirectional layers).
* ``transformer``  -- pre-LN self-attention encoder (2 layers, FFN x4).
  NOTE: no key-padding mask on purpose -- the Mamba stack scans padding
  tokens too, so masking only the transformer would be an unfair advantage.
* ``gated_mlp``    -- per-token MLP, **no token mixing** (lower bound that
  shows sequence mixing is needed at all).
"""

from __future__ import annotations

from dataclasses import replace

import torch.nn as nn
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm
from torch import Tensor

from .config import Mamba3StackConfig, MixerConfig
from .mamba3 import Mamba3
from .mamba3_stack import Mamba3Stack


class BidirMamba3Stack(nn.Module):
    """Bidirectional Mamba-3: per layer fw + reversed bw scans, averaged.

    ``x = x + dropout(0.5 * (fw(norm(x)) + flip(bw(flip(norm(x))))))``

    Direction-specific weights; ``layer_idx``/``n_layer`` count every
    directional block so the depth-dependent init scaling matches the
    unidirectional stack of equivalent total depth.
    """

    def __init__(self, cfg: Mamba3StackConfig):
        super().__init__()
        self.cfg = cfg
        n_total = cfg.n_layers * 2  # directional blocks in the residual stream

        def _block(idx: int) -> Mamba3:
            return Mamba3(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                expand=cfg.expand,
                headdim=cfg.headdim,
                ngroups=cfg.ngroups,
                chunk_size=cfg.chunk_size,
                is_mimo=cfg.is_mimo,
                mimo_rank=cfg.mimo_rank,
                is_outproj_norm=cfg.is_outproj_norm,
                layer_idx=idx,
                n_layer=n_total,
            )

        self.fw = nn.ModuleList(_block(2 * i) for i in range(cfg.n_layers))
        self.bw = nn.ModuleList(_block(2 * i + 1) for i in range(cfg.n_layers))
        self.norms = nn.ModuleList(RMSNorm(cfg.d_model, eps=1e-5) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, eps=1e-5)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3, f"x must be (B, L, d_model); got {tuple(x.shape)}"
        for norm, fw, bw in zip(self.norms, self.fw, self.bw):
            h = norm(x)
            y_fw = fw(h)
            y_bw = bw(h.flip(1).contiguous()).flip(1)
            x = x + self.dropout(0.5 * (y_fw + y_bw))
        return self.final_norm(x)


class TransformerStack(nn.Module):
    """Pre-LN self-attention encoder, param-matched to the Mamba-3 stack."""

    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3, f"x must be (B, L, d_model); got {tuple(x.shape)}"
        return self.final_norm(self.encoder(x))


class GatedMLPStack(nn.Module):
    """Per-token MLP residual stack -- intentionally **no token mixing**."""

    def __init__(self, *, d_model: int, n_layers: int, hidden: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3, f"x must be (B, L, d_model); got {tuple(x.shape)}"
        for layer in self.layers:
            x = x + self.dropout(layer(x))
        return self.final_norm(x)


def build_mixer(mx: MixerConfig, m3: Mamba3StackConfig) -> nn.Module:
    """Build the token mixer selected by ``mx.kind`` (param-matched depths)."""
    n = mx.resolved_layers
    if mx.kind == "mamba3":
        return Mamba3Stack(replace(m3, n_layers=n))
    if mx.kind == "mamba3_bidir":
        return BidirMamba3Stack(replace(m3, n_layers=n))
    if mx.kind == "transformer":
        return TransformerStack(
            d_model=m3.d_model,
            n_layers=n,
            n_heads=mx.n_heads,
            ffn_mult=mx.ffn_mult,
            dropout=m3.dropout,
        )
    if mx.kind == "gated_mlp":
        return GatedMLPStack(d_model=m3.d_model, n_layers=n, hidden=3072, dropout=m3.dropout)
    raise ValueError(f"unknown mixer kind: {mx.kind!r}")


__all__ = [
    "BidirMamba3Stack",
    "GatedMLPStack",
    "TransformerStack",
    "build_mixer",
]
