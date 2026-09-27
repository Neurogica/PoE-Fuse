"""Stack of Mamba-3 blocks with pre-norm residual connections.

The Mamba-3 block itself lives in :mod:`poe_fuse.mamba3` (a vendored
copy from Goombalab's ``mamba_ssm``).  This module just wires N of those
blocks together with the standard pre-norm residual layout the SSM language
modelling literature uses::

    x = x + Mamba3(RMSNorm(x))

so the configured ``d_model`` matches the codec's ``d_s``.
"""

from __future__ import annotations

import torch.nn as nn
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm
from torch import Tensor

from .config import Mamba3StackConfig
from .mamba3 import Mamba3


class Mamba3Stack(nn.Module):
    def __init__(self, cfg: Mamba3StackConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                Mamba3(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    expand=cfg.expand,
                    headdim=cfg.headdim,
                    ngroups=cfg.ngroups,
                    chunk_size=cfg.chunk_size,
                    is_mimo=cfg.is_mimo,
                    mimo_rank=cfg.mimo_rank,
                    is_outproj_norm=cfg.is_outproj_norm,
                    layer_idx=i,
                    n_layer=cfg.n_layers,
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.norms = nn.ModuleList([RMSNorm(cfg.d_model, eps=1e-5) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, eps=1e-5)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3, f"x must be (B, L, d_model); got {tuple(x.shape)}"
        for norm, block in zip(self.norms, self.layers):
            x = x + self.dropout(block(norm(x)))
        return self.final_norm(x)


__all__ = ["Mamba3Stack"]
