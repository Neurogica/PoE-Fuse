"""PoEFuse top-level model: trio-backbone codec + task head dispatcher.

``PoEFuse(cfg)`` builds the trio-backbone codec (DINOv3 ViT-7B/16 + SAM 3
image model + Gemma-4-E4B-it LM) and a single task head selected by
``cfg.head.kind``:

* ``classifier``    -> 5-way xBD damage classifier (returns ``logits``).
* ``bbox_grid``     -> grid-anchor detector (returns ``obj_logits`` and
                       ``bbox_pred`` in TEOChatlas 0-100 coordinates).

The forward signature is
``forward(images, questions, sam3_prompts, **task_inputs)``.  The two text
inputs are per-sample lists of strings of length ``B``; ``questions`` is
fed to the Gemma-4 LM as the user turn while ``sam3_prompts`` is the SAM 3
text prompt (typically a short noun phrase such as ``"building"``).
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from .codec import PoEFuseCodec
from .config import PoEFuseConfig
from .heads import (
    BBoxGridHead,
    ClassifierHead,
    GemmaQADecoderHead,
    build_head,
)


class PoEFuse(nn.Module):
    def __init__(self, cfg: PoEFuseConfig, *, build_backbones: bool = True):
        super().__init__()
        self.cfg = cfg
        self.codec = PoEFuseCodec(cfg, build_backbones=build_backbones)
        self.head = build_head(cfg.head, d_model=cfg.d_s)
        self.adapter: nn.Module | None = None

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (not trainable_only) or p.requires_grad)

    def _dispatch_head(
        self,
        codec_out: dict[str, Any],
        *,
        labels: Tensor | None,
        bbox_targets: list[list[list[float]]] | None,
        text_ids: Tensor | None,
    ) -> dict[str, Any]:
        tokens = codec_out["tokens"]
        token_mask = codec_out["token_mask"]
        if isinstance(self.head, ClassifierHead):
            return self.head(
                tokens,
                labels=labels,
                token_mask=token_mask,
                transport_summary=codec_out.get("transport_summary"),
            )
        if isinstance(self.head, BBoxGridHead):
            targets = None
            if bbox_targets is not None:
                targets = [[tuple(b) for b in sample] for sample in bbox_targets]
            return self.head(tokens, bbox_targets=targets, token_mask=token_mask)
        if isinstance(self.head, GemmaQADecoderHead):
            return self.head(tokens, text_ids=text_ids, token_mask=token_mask)
        raise TypeError(f"unsupported head type: {type(self.head).__name__}")

    def forward_features(
        self,
        raw: dict[str, Tensor],
        *,
        labels: Tensor | None = None,
        bbox_targets: list[list[list[float]]] | None = None,
        text_ids: Tensor | None = None,
    ) -> dict[str, Any]:
        """Head forward from pre-computed (cached) backbone features.

        ``raw`` is the dict produced by
        :meth:`PoEFuseCodec.encode_backbones` (or loaded from the on-disk
        cache): ``dino``/``sam3``/``gemma`` ``(B, T, N, d)`` + ``image_mask``.
        """
        codec_out = self.codec.mix(raw)
        if self.adapter is not None:
            codec_out = {
                **codec_out,
                "tokens": self.adapter(codec_out["tokens"]),
            }
        return self._dispatch_head(
            codec_out, labels=labels, bbox_targets=bbox_targets, text_ids=text_ids
        )

    def forward(
        self,
        images: Tensor,
        questions: list[str],
        sam3_prompts: list[str],
        *,
        image_mask: Tensor | None = None,
        labels: Tensor | None = None,
        bbox_targets: list[list[list[float]]] | None = None,
        text_ids: Tensor | None = None,
    ) -> dict[str, Any]:
        codec_out = self.codec(
            images,
            questions,
            sam3_prompts,
            image_mask=image_mask,
        )
        if self.adapter is not None:
            codec_out = {
                **codec_out,
                "tokens": self.adapter(codec_out["tokens"]),
            }
        return self._dispatch_head(
            codec_out, labels=labels, bbox_targets=bbox_targets, text_ids=text_ids
        )


__all__ = ["PoEFuse"]
