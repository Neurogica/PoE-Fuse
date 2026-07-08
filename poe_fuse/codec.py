"""PoEFuse trio-backbone codec.

The codec replaces the previous DINOv2 + SigLIP + HMSS pipeline.  Given an
image-pair stack ``images: (B, T=2, 3, H, W)`` plus per-sample text inputs
(one ``question`` and one ``sam3_prompt`` per row in the batch) it produces
a single token stream ``(B, L_vis, d_s)`` plus a matching boolean
``token_mask``.

The forward path is split into two stages so the expensive frozen part can
be **precomputed once and cached to disk** (see ``feature_cache.py`` and
``scripts/cache_features.py``):

* :meth:`PoEFuseCodec.encode_backbones` -- runs the three frozen foundation
  models (DINOv3 ViT-7B/16, SAM 3 image model, Gemma-4-E4B-it) and returns
  their raw penultimate features.  This is the heavy, parameter-frozen step.
* :meth:`PoEFuseCodec.mix` -- the only *trainable* part: per-branch ``Linear``
  projections to ``d_s``, sequence concat, then the Mamba-3 stack.

Pipeline per time step ``t`` in ``[0, T)`` (inside ``encode_backbones``)::

    img_t = images[:, t]                              (B, 3, H, W)
    dino_t = DinoV3PenultimateHook(img_t)             (B, 196, 4096)
    sam3_t = Sam3PenultimateHook(img_t, sam3_prompt)  (B, 200, 256)
    gem_t  = Gemma4PenultimateHook(img_t, question)   (B, 256, 2560)

``mix`` then projects + concatenates + runs Mamba-3::

    p_dino = LinearDino(dino)   p_sam3 = LinearSam3(sam3)   p_gem = LinearGemma(gem)
    Z = concat([p_dino, p_sam3, p_gem], dim=token)        (B, T*676, d_s)
    H = Mamba3Stack(Z)                                    (B, T*676, d_s)

When ``build_backbones=False`` the three frozen models are *not* constructed
(neither their weights nor the ``dinov3`` / ``sam3`` / ``transformers``
imports are required).  This lets cached-feature training run on a host that
only has the small trainable modules + ``mamba_ssm`` available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .config import PoEFuseConfig
from .fusion import CrossExpertChangeFusion, SpatialChangeAgreementFusion
from .mamba3_stack import Mamba3Stack
from .mixers import build_mixer

# Keys used both by ``encode_backbones`` outputs and the on-disk cache.
BRANCH_KEYS: tuple[str, ...] = ("dino", "sam3", "gemma")


class PoEFuseCodec(nn.Module):
    """Trio-backbone (frozen) + per-branch Linear + Mamba3 stack (trainable)."""

    def __init__(self, cfg: PoEFuseConfig, *, build_backbones: bool = True):
        super().__init__()
        self.cfg = cfg
        bb = cfg.backbone
        self.build_backbones = build_backbones

        if build_backbones:
            # Imported lazily so cached-feature training need not have the
            # heavy ``dinov3`` / ``sam3`` / multimodal ``transformers`` stack.
            from .backbones import (
                DinoV3PenultimateHook,
                Gemma4PenultimateHook,
                Sam3PenultimateHook,
            )

            self.dinov3 = DinoV3PenultimateHook(bb)
            self.sam3 = Sam3PenultimateHook(bb)
            self.gemma4 = Gemma4PenultimateHook(bb)
        else:
            self.dinov3 = None
            self.sam3 = None
            self.gemma4 = None

        d_s = cfg.d_s
        self.proj_dinov3 = nn.Linear(bb.dinov3_hidden_size, d_s, bias=False)
        self.proj_sam3 = nn.Linear(bb.sam3_hidden_size, d_s, bias=False)
        self.proj_gemma4 = nn.Linear(bb.gemma4_hidden_size, d_s, bias=False)

        # Sequence mixer.  Keep the legacy attribute name
        # ``mamba3`` so default configs stay state-dict compatible with old
        # checkpoints (default MixerConfig builds the identical Mamba3Stack).
        mx = cfg.mixer
        if mx.kind == "mamba3" and mx.resolved_layers == cfg.mamba3.n_layers:
            self.mamba3 = Mamba3Stack(cfg.mamba3)
        else:
            self.mamba3 = build_mixer(mx, cfg.mamba3)
        self.scan_order = mx.scan_order
        self._scan_perm: Tensor | None = None  # cached permutation index

        self.fusion_cfg = cfg.fusion
        raw_active = cfg.fusion.active_experts or BRANCH_KEYS
        if isinstance(raw_active, list):
            raw_active = tuple(raw_active)
        active = tuple(k for k in BRANCH_KEYS if k in raw_active)
        if not active:
            raise ValueError(f"fusion.active_experts must be non-empty subset of {BRANCH_KEYS}")
        self.active_keys = active
        self._n_per_image_active = sum(
            bb.dinov3_n_tokens
            if k == "dino"
            else bb.sam3_n_tokens
            if k == "sam3"
            else bb.gemma4_n_image_tokens
            for k in active
        )

        if cfg.fusion.enabled and cfg.fusion.spatial_align:
            # Headline path: heterogeneous experts -> shared spatial grid (2)
            # + per-location cross-expert change agreement (1).
            self.fusion = SpatialChangeAgreementFusion(cfg.fusion, d_model=d_s, branch_keys=active)
        elif cfg.fusion.enabled:
            self.fusion = CrossExpertChangeFusion(cfg.fusion, d_model=d_s, branch_keys=active)
        else:
            self.fusion = None

        self._n_per_image = bb.dinov3_n_tokens + bb.sam3_n_tokens + bb.gemma4_n_image_tokens

    @property
    def n_per_image(self) -> int:
        return self._n_per_image

    # ------------------------------------------------------------------ #
    # Stage 1 (frozen, expensive, cacheable)
    # ------------------------------------------------------------------ #
    def encode_backbones(
        self,
        images: Tensor,
        questions: list[str],
        sam3_prompts: list[str],
        *,
        image_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run the three frozen backbones and return raw penultimate features.

        Returns a dict with one ``(B, T, N_branch, d_branch)`` tensor per
        branch (``"dino"``, ``"sam3"``, ``"gemma"``) plus ``"image_mask"``
        ``(B, T)``.  All feature tensors are float32 (matching the hooks).
        """
        if not self.build_backbones:
            raise RuntimeError(
                "encode_backbones() called on a codec built with "
                "build_backbones=False; load cached features and call mix() instead"
            )
        assert images.dim() == 5, f"images must be (B, T, 3, H, W); got {tuple(images.shape)}"
        B, T, _C, _H, _W = images.shape
        if len(questions) != B or len(sam3_prompts) != B:
            raise ValueError(
                f"questions/sam3_prompts must have length B={B}; got "
                f"{len(questions)} / {len(sam3_prompts)}"
            )
        if image_mask is None:
            image_mask = images.new_ones(B, T, dtype=torch.bool)
        else:
            assert image_mask.shape == (B, T), (
                f"image_mask must be (B={B}, T={T}); got {tuple(image_mask.shape)}"
            )
            image_mask = image_mask.to(torch.bool)

        dino_t, sam3_t, gem_t = [], [], []
        for t in range(T):
            img_t = images[:, t]
            dino_t.append(self.dinov3(img_t))
            sam3_t.append(self.sam3(img_t, sam3_prompts))
            gem_t.append(self.gemma4(img_t, questions))

        return {
            "dino": torch.stack(dino_t, dim=1),  # (B, T, 196, 4096)
            "sam3": torch.stack(sam3_t, dim=1),  # (B, T, 200, 256)
            "gemma": torch.stack(gem_t, dim=1),  # (B, T, 256, 2560)
            "image_mask": image_mask,  # (B, T)
        }

    # ------------------------------------------------------------------ #
    # Stage 2 (trainable)
    # ------------------------------------------------------------------ #
    def mix(self, raw: dict[str, Tensor]) -> dict[str, Tensor]:
        """Project + concat + Mamba-3 over raw (or cached) backbone features.

        ``raw`` must provide ``"dino"``/``"sam3"``/``"gemma"`` shaped
        ``(B, T, N_branch, d_branch)`` and ``"image_mask"`` ``(B, T)``.
        """
        dino = raw["dino"]
        sam3 = raw["sam3"]
        gemma = raw["gemma"]
        image_mask = raw["image_mask"]
        if dino.dim() != 4:
            raise ValueError(f"expected (B, T, N, d) dino features; got {tuple(dino.shape)}")
        B, T = dino.shape[0], dino.shape[1]

        # Linear layers act on the last dim; cast cached fp16 -> projection dtype.
        proj_dtype = self.proj_dinov3.weight.dtype
        p_dino = self.proj_dinov3(dino.to(proj_dtype))
        p_sam3 = self.proj_sam3(sam3.to(proj_dtype))
        p_gem = self.proj_gemma4(gemma.to(proj_dtype))
        image_mask = image_mask.to(torch.bool)

        proj = {
            k: v
            for k, v in (("dino", p_dino), ("sam3", p_sam3), ("gemma", p_gem))
            if k in self.active_keys
        }

        if self.fusion is not None:
            # Cross-Expert Change-Agreement fusion (+ explicit difference path).
            tokens, token_mask = self.fusion(proj, image_mask)
        else:
            # Original path: concat active experts over the token axis, flatten time.
            z = torch.cat([proj[k] for k in self.active_keys], dim=2)  # (B, T, N, d_s)
            n_per = z.shape[2]
            tokens = z.reshape(B, T * n_per, z.size(-1))
            token_mask = image_mask.unsqueeze(-1).expand(B, T, n_per).reshape(B, T * n_per)

        # Spatial change-field for the dense change-seg head (B, G, G, d).
        change_grid = self._change_grid(proj, B)

        if self.scan_order == "location_major":
            # Change-Anchored Scan: interleave the aligned grids so
            # each location's (t0, t1, change) tokens are adjacent.  All heads
            # are order-invariant (masked mean / attention pooling), so no
            # inverse permutation is needed.
            perm = self._location_major_perm(tokens.shape[1], tokens.device)
            tokens = tokens[:, perm]
            token_mask = token_mask[:, perm]

        tokens = self.mamba3(tokens.contiguous())
        return {
            "tokens": tokens,
            "token_mask": token_mask,
            "change_grid": change_grid,
            "transport_summary": getattr(self.fusion, "last_transport_summary", None),
        }

    def _location_major_perm(self, L: int, device: torch.device) -> Tensor:
        """Permutation turning the segment-major aligned stream into
        location-major order: ``(t0_i, t1_i, ..., change_i)`` adjacent per grid
        location ``i``; trailing summary tokens stay at the end."""
        if self._scan_perm is not None and self._scan_perm.numel() == L:
            return self._scan_perm.to(device)
        layout = getattr(self.fusion, "last_layout", None)
        if layout is None:
            raise RuntimeError(
                "location_major scan requires SpatialChangeAgreementFusion "
                "(fusion.spatial_align=true); no layout was recorded"
            )
        g2 = layout["n_grid"]
        n_seg = layout["n_time"] + (1 if layout["has_change"] else 0)
        base = torch.arange(g2, device=device)
        cols = [s * g2 + base for s in range(n_seg)]  # one (g2,) per segment
        perm = torch.stack(cols, dim=1).reshape(-1)  # interleaved (n_seg*g2,)
        tail = torch.arange(n_seg * g2, L, device=device)  # summaries untouched
        perm = torch.cat([perm, tail])
        if perm.numel() != L:
            raise RuntimeError(f"scan permutation covers {perm.numel()} tokens but stream has {L}")
        self._scan_perm = perm
        return perm

    def _change_grid(self, proj: dict[str, Tensor], B: int) -> Tensor:
        """Return the ``(B, G, G, d)`` spatial change field for the seg head."""
        fusion = self.fusion
        if (
            isinstance(fusion, SpatialChangeAgreementFusion)
            and getattr(fusion, "last_change_grid", None) is not None
        ):
            cg = fusion.last_change_grid  # (B, Gf*Gf, d)
            gf = fusion.G
            return cg.view(B, gf, gf, cg.size(-1))

        # No spatial fusion: build a coarse grid from the active expert(s).
        # Prefer DINOv3 (native 14x14 patches); otherwise use the sole expert.
        g = int(round(self.cfg.backbone.dinov3_n_tokens**0.5))
        if "dino" in proj:
            dd = proj["dino"][:, -1] - proj["dino"][:, 0]  # (B, 196, d)
            return dd.view(B, g, g, dd.size(-1))
        if len(self.active_keys) == 1:
            key = self.active_keys[0]
            dd = proj[key][:, -1] - proj[key][:, 0]  # (B, N, d)
            n = dd.shape[1]
            side = int(round(n**0.5))
            if side * side == n:
                grid = dd.view(B, side, side, dd.size(-1))
            else:
                # Non-square token sets (e.g. SAM3 queries): bilinear to GxG.
                x = dd.transpose(1, 2).unsqueeze(2)  # (B, d, 1, N)
                x = torch.nn.functional.interpolate(
                    x, size=(g, g), mode="bilinear", align_corners=False
                )
                grid = x.squeeze(2).permute(0, 2, 3, 1)  # (B, G, G, d)
            return grid
        # Multi-expert concat path without spatial fusion: DINO diff as proxy grid.
        dd = proj["dino"][:, -1] - proj["dino"][:, 0] if "dino" in proj else None
        if dd is None:
            # No DINO: average diffs from active experts then tile to GxG.
            parts = [proj[k][:, -1] - proj[k][:, 0] for k in self.active_keys]
            pooled = torch.stack([p.mean(dim=1) for p in parts], dim=0).mean(dim=0)
            return pooled[:, None, None, :].expand(B, g, g, pooled.size(-1))
        return dd.view(B, g, g, dd.size(-1))

    # ------------------------------------------------------------------ #
    # Convenience: full path (encode + mix)
    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: Tensor,
        questions: list[str],
        sam3_prompts: list[str],
        *,
        image_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        raw = self.encode_backbones(images, questions, sam3_prompts, image_mask=image_mask)
        return self.mix(raw)


__all__ = ["PoEFuseCodec", "BRANCH_KEYS"]
