"""Task-specific prediction heads on top of PoEFuse visual tokens.

Three heads cover the four target tasks ():

* ``ClassifierHead``      -- xbd_dmg_cls   (5-way classification).
* ``BBoxGridHead``        -- s2_det / xbd_loc (1-stage anchor detector).
* ``TextDecoderHead``     -- s2looking_sre (char-level autoregressive
                              text generation, also covers Yes/No QA).

All heads share the same input contract: ``tokens`` is the codec output
``(B, L_vis, d_model)``.  They emit their own loss when ``labels`` are
provided and a simple decoded representation at inference time.
"""

from __future__ import annotations

import string
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

from .config import HeadConfig

# -- character vocab for the text decoder --------------------------------

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
_PRINTABLE = string.printable
TEXT_VOCAB: list[str] = ["<pad>", "<bos>", "<eos>"] + list(_PRINTABLE)
TEXT_VOCAB_SIZE: int = len(TEXT_VOCAB)
_CHAR_TO_ID: dict[str, int] = {ch: i for i, ch in enumerate(TEXT_VOCAB)}


def encode_text(text: str, *, max_len: int) -> list[int]:
    """Return ``[BOS, c1, c2, ..., EOS]`` truncated to ``max_len`` ids."""
    ids = [BOS_ID]
    for ch in text:
        token = _CHAR_TO_ID.get(ch)
        if token is None:
            continue
        ids.append(token)
        if len(ids) >= max_len - 1:
            break
    ids.append(EOS_ID)
    return ids


def decode_text(ids: list[int]) -> str:
    chars: list[str] = []
    for tok in ids:
        if tok in (PAD_ID, BOS_ID):
            continue
        if tok == EOS_ID:
            break
        if 0 <= tok < TEXT_VOCAB_SIZE and tok > EOS_ID:
            chars.append(TEXT_VOCAB[tok])
    return "".join(chars)


# -- heads ----------------------------------------------------------------


class ClassifierHead(nn.Module):
    """Linear classifier on pooled visual tokens."""

    def __init__(self, d_model: int, *, num_classes: int, pool: str, focal_gamma: float = 0.0):
        super().__init__()
        if pool not in {"mean", "cls"}:
            raise ValueError(f"pool must be 'mean' or 'cls'; got {pool!r}")
        self.pool = pool
        self.num_classes = num_classes
        self.focal_gamma = float(focal_gamma)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        if pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # (v12) head-aware transport readout: a pooled 8-stat OT summary is added
        # to the *pooled* features through a dedicated gate, so the fragile
        # classifier token stream / change field is NEVER perturbed (the v11b
        # dmg_cls collapse fix).  ts_gate init 0 => logits bit-identical at init
        # (basin-safe); ts_lift.weight std 1e-2 (NON-zero) => live gradient to the
        # gate from step 0, so it is safe AND learnable (not the dead-grad trap).
        self.ts_norm = nn.LayerNorm(8)
        self.ts_lift = nn.Linear(8, d_model)
        nn.init.normal_(self.ts_lift.weight, std=1e-2)
        nn.init.zeros_(self.ts_lift.bias)
        self.ts_gate = nn.Parameter(torch.zeros(()))
        # Optional per-class loss weights (inverse-frequency) used to fight
        # the xBD damage-classification collapse to the majority class.
        # Registered as a buffer so it moves with ``.to(device)`` and is
        # saved/restored with the state dict.
        self.register_buffer("class_weights", None, persistent=False)

    def set_class_weights(self, weights: Tensor | None) -> None:
        """Install per-class CE weights (``None`` disables weighting)."""
        if weights is None:
            self.class_weights = None
            return
        w = weights.detach().float().flatten()
        if w.numel() != self.num_classes:
            raise ValueError(f"class_weights must have {self.num_classes} entries; got {w.numel()}")
        self.class_weights = w.to(self.fc.weight.device)

    def _pool(self, tokens: Tensor, token_mask: Tensor | None) -> Tensor:
        if self.pool == "mean":
            if token_mask is None:
                return tokens.mean(dim=1)
            # Mean over valid tokens only (avoid diluting with zero-pads).
            mask = token_mask.to(tokens.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (tokens * mask).sum(dim=1) / denom
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        cat_tokens = torch.cat([cls, tokens], dim=1)
        if token_mask is None:
            return cat_tokens.mean(dim=1)
        cls_mask = token_mask.new_ones(token_mask.size(0), 1)
        cat_mask = torch.cat([cls_mask, token_mask], dim=1).to(tokens.dtype).unsqueeze(-1)
        denom = cat_mask.sum(dim=1).clamp_min(1.0)
        return (cat_tokens * cat_mask).sum(dim=1) / denom

    def forward(
        self,
        tokens: Tensor,
        *,
        labels: Tensor | None = None,
        token_mask: Tensor | None = None,
        transport_summary: Tensor | None = None,
    ) -> dict:
        pooled = self.norm(self._pool(tokens, token_mask))
        if transport_summary is not None:
            # tanh(ts_gate=0)=0 => exact no-op at init; gradient still flows.
            pooled = pooled + torch.tanh(self.ts_gate) * self.ts_lift(
                self.ts_norm(transport_summary.to(pooled.dtype))
            )
        logits = self.fc(pooled)
        out = {"logits": logits, "predictions": logits.argmax(dim=-1)}
        if labels is not None:
            weight = self.class_weights
            if weight is not None:
                weight = weight.to(logits.device, logits.dtype)
            if self.focal_gamma > 0:
                # Focal-CE: (1-p_t)^gamma * weighted-CE. Keeps inverse-freq
                # weights (rare-class boost) but down-weights the easy majority
                # (no-damage) that leaks into minor -- the verified bottleneck.
                ce = F.cross_entropy(logits, labels.long(), weight=weight, reduction="none")
                pt = torch.exp(-F.cross_entropy(logits, labels.long(), reduction="none"))
                out["loss"] = ((1.0 - pt) ** self.focal_gamma * ce).mean()
            else:
                out["loss"] = F.cross_entropy(logits, labels.long(), weight=weight)
        return out


class BBoxGridHead(nn.Module):
    """One-stage anchor head producing ``grid_side**2`` candidate bboxes.

    Each grid cell predicts an objectness logit and a normalised bbox
    ``(x1, y1, x2, y2)`` in ``[0, 1]`` via sigmoid; coordinates are
    rescaled to ``[0, 100]`` (matching the TEOChatlas ground-truth scale)
    when decoded.

    Training: assign each GT bbox to the cell whose center contains the
    GT bbox center.  Loss = focal(obj) + lambda * L1(bbox) on positives.
    The objectness branch uses ``torchvision.ops.sigmoid_focal_loss`` to
    handle the heavy negative-cell imbalance (especially s2_det, where a
    plain BCE collapses the head to all-negatives).

    No Hungarian matching, no per-cell anchor priors -- the simplest
    1-stage detector that is well-defined for this task.
    """

    def __init__(
        self,
        d_model: int,
        *,
        grid_side: int,
        lambda_box: float,
        lambda_obj: float,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.grid_side = grid_side
        self.num_cells = grid_side * grid_side
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.norm = nn.LayerNorm(d_model)
        self.cell_queries = nn.Parameter(torch.randn(1, self.num_cells, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 5),
        )

    def _predict(self, tokens: Tensor, token_mask: Tensor | None) -> Tensor:
        """Return ``(B, grid^2, 5)`` -> ``(obj_logit, x1, y1, x2, y2)``.

        Bbox channels go through sigmoid in the decoder, not here, so the
        loss can use ``sigmoid_focal_loss`` on the raw objectness logit.
        """
        tokens = self.norm(tokens)
        q = self.cell_queries.expand(tokens.size(0), -1, -1)
        kpm = None if token_mask is None else ~token_mask.to(torch.bool)
        attn_out, _ = self.attn(q, tokens, tokens, key_padding_mask=kpm, need_weights=False)
        return self.mlp(attn_out)

    def forward(
        self,
        tokens: Tensor,
        *,
        bbox_targets: list[list[tuple[float, float, float, float]]] | None = None,
        token_mask: Tensor | None = None,
    ) -> dict:
        raw = self._predict(tokens, token_mask)  # (B, grid^2, 5)
        obj_logits = raw[..., 0]
        bbox_pred = raw[..., 1:].sigmoid() * 100.0  # to TEOChatlas 0-100 scale

        out: dict = {"obj_logits": obj_logits, "bbox_pred": bbox_pred}
        if bbox_targets is None:
            return out

        device = raw.device
        raw.size(0)
        obj_target = torch.zeros_like(obj_logits)
        bbox_target = torch.zeros_like(bbox_pred)
        positive_mask = torch.zeros_like(obj_logits, dtype=torch.bool)

        for b, gts in enumerate(bbox_targets):
            for box in gts:
                x1, y1, x2, y2 = box
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                cell_x = min(self.grid_side - 1, max(0, int(cx * self.grid_side / 100.0)))
                cell_y = min(self.grid_side - 1, max(0, int(cy * self.grid_side / 100.0)))
                idx = cell_y * self.grid_side + cell_x
                obj_target[b, idx] = 1.0
                bbox_target[b, idx] = torch.tensor(box, device=device, dtype=bbox_target.dtype)
                positive_mask[b, idx] = True

        loss_obj = sigmoid_focal_loss(
            obj_logits,
            obj_target,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="mean",
        )
        if positive_mask.any():
            loss_box = F.l1_loss(bbox_pred[positive_mask], bbox_target[positive_mask]) / 100.0
        else:
            loss_box = obj_logits.new_zeros(())
        loss = self.lambda_obj * loss_obj + self.lambda_box * loss_box
        out["loss"] = loss
        out["loss_obj"] = loss_obj.detach()
        out["loss_box"] = loss_box.detach()
        return out

    @torch.inference_mode()
    def decode(
        self,
        tokens: Tensor,
        *,
        threshold: float = 0.5,
        token_mask: Tensor | None = None,
    ) -> list[list[list[float]]]:
        """Per-sample list of ``[x1, y1, x2, y2]`` with ``obj_prob > threshold``."""
        raw = self._predict(tokens, token_mask)
        obj_prob = raw[..., 0].sigmoid()
        bbox = raw[..., 1:].sigmoid() * 100.0
        results: list[list[list[float]]] = []
        for b in range(raw.size(0)):
            keep = (obj_prob[b] > threshold).nonzero(as_tuple=False).flatten().tolist()
            results.append([[round(float(c), 1) for c in bbox[b, i].tolist()] for i in keep])
        return results


class TextDecoderHead(nn.Module):
    """Character-level autoregressive decoder cross-attending over visual tokens.

    Vocab is fixed (``TEXT_VOCAB``): ``<pad>``, ``<bos>``, ``<eos>`` and
    Python's ``string.printable`` (=> 103 ids).  Training uses teacher
    forcing on the gold answer string; inference greedy-decodes up to
    ``max_len`` tokens.
    """

    def __init__(
        self,
        d_model: int,
        *,
        vocab_size: int,
        max_len: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        if vocab_size != TEXT_VOCAB_SIZE:
            raise ValueError(
                f"TextDecoderHead expects vocab_size={TEXT_VOCAB_SIZE} to match the fixed "
                f"character vocab; got {vocab_size}"
            )
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.memory_norm = nn.LayerNorm(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.zeros_(self.fc.bias) if self.fc.bias is not None else None

    def _decode_step(
        self,
        ids: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None,
    ) -> Tensor:
        T = ids.size(1)
        emb = self.tok_emb(ids) + self.pos_emb[:, :T]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=ids.device), diagonal=1)
        out = self.decoder(
            tgt=emb,
            memory=memory,
            tgt_mask=causal,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc(self.norm(out))

    def forward(
        self,
        tokens: Tensor,
        *,
        text_ids: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> dict:
        memory = self.memory_norm(tokens)
        memory_kpm = None if token_mask is None else ~token_mask.to(torch.bool)
        if text_ids is None:
            return {"memory": memory}
        inputs = text_ids[:, :-1]
        targets = text_ids[:, 1:]
        logits = self._decode_step(inputs, memory, memory_kpm)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_ID,
        )
        return {"loss": loss, "logits": logits}

    @torch.inference_mode()
    def generate(self, tokens: Tensor, *, token_mask: Tensor | None = None) -> list[str]:
        memory = self.memory_norm(tokens)
        memory_kpm = None if token_mask is None else ~token_mask.to(torch.bool)
        B = tokens.size(0)
        device = tokens.device
        ids = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        for _ in range(self.max_len - 1):
            logits = self._decode_step(ids, memory, memory_kpm)
            next_id = logits[:, -1].argmax(dim=-1)
            next_id = torch.where(done, torch.full_like(next_id, PAD_ID), next_id)
            ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
            done = done | (next_id == EOS_ID)
            if bool(done.all()):
                break
        return [decode_text(ids[b].tolist()) for b in range(B)]


class ChangeSegHead(nn.Module):
    """Dense change-segmentation head (native S2Looking task).

    Takes the codec's **spatial change field** ``(B, G, G, d)`` -- for the
    shared-grid fusion this is the agreement-gated cross-expert change map; for
    the baseline it is the DINOv3 patch-difference grid -- and convolutionally
    decodes it to an ``R x R`` per-pixel change-mask, bilinearly upsampled to
    the evaluation resolution.  A conv decoder gives the spatial inductive bias
    that makes dense masks learnable (unlike attention over unordered tokens),
    and it is *identical* across ablations, so any gain comes from the quality
    of the input change field -- exactly the fusion contribution we measure.

    Loss = positive-weighted ``BCEWithLogits`` + soft Dice at ``R x R``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        out_size: int,
        num_heads: int = 4,  # kept for config compatibility (unused by conv)
        num_layers: int = 2,
        lambda_dice: float = 1.0,
        pos_weight: float = 5.0,
        hidden: int = 256,
    ):
        super().__init__()
        self.out_size = out_size
        self.lambda_dice = lambda_dice
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.in_norm = nn.LayerNorm(d_model)
        self.proj = nn.Conv2d(d_model, hidden, kernel_size=1)
        body = []
        for _ in range(max(1, num_layers)):
            body += [
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*body)
        self.out_conv = nn.Conv2d(hidden, 1, kernel_size=1)

    def _predict(self, grid: Tensor) -> Tensor:
        """``grid`` is ``(B, G, G, d)`` -> ``(B, R, R)`` change logits."""
        x = self.in_norm(grid).permute(0, 3, 1, 2).contiguous()  # (B, d, G, G)
        x = self.proj(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(
                x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
            )
        x = self.body(x)
        return self.out_conv(x).squeeze(1)  # (B, R, R)

    def forward(
        self,
        grid: Tensor,
        *,
        target_mask: Tensor | None = None,
        **_ignore,
    ) -> dict:
        logits = self._predict(grid)  # (B, R, R)
        out: dict = {"mask_logits": logits}
        if target_mask is None:
            return out
        target = target_mask.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight.to(logits.dtype)
        )
        prob = logits.sigmoid()
        dims = (1, 2)
        inter = (prob * target).sum(dim=dims)
        denom = prob.sum(dim=dims) + target.sum(dim=dims)
        dice = 1.0 - ((2 * inter + 1.0) / (denom + 1.0)).mean()
        out["loss"] = bce + self.lambda_dice * dice
        out["loss_bce"] = bce.detach()
        out["loss_dice"] = dice.detach()
        return out

    @torch.inference_mode()
    def decode(
        self,
        grid: Tensor,
        *,
        eval_size: int = 256,
        threshold: float = 0.5,
        **_ignore,
    ) -> Tensor:
        """Return ``(B, eval_size, eval_size)`` binary change masks (uint8)."""
        logits = self._predict(grid)
        up = F.interpolate(
            logits.unsqueeze(1), size=(eval_size, eval_size), mode="bilinear", align_corners=False
        ).squeeze(1)
        return (up.sigmoid() > threshold).to(torch.uint8)


class ReferringSegHead(nn.Module):
    """Referring-expression segmentation head (native S2Looking SRE task).

    SRE must localise the region named by a *referring expression*.  Early
    attempts that let an ``R x R`` grid of query cells freely cross-attend over
    the unordered token stream could not localise at all -- with no spatial
    inductive bias they collapse to a constant (all-positive) mask, even when
    overfitting a handful of samples.

    This head instead reuses the **conv-on-grid** recipe that makes
    :class:`ChangeSegHead` learnable, and turns it into a *referring* decoder
    by FiLM-conditioning the convolutional features on the referring
    expression.  The spatial backbone is the codec's change field
    ``(B, G, G, d)`` (spatially ordered, so a conv has a real spatial prior);
    the text -- already encoded as question-conditioned Gemma soft tokens in
    the codec token stream -- is attention-pooled to a single vector that
    produces per-channel ``(gamma, beta)`` modulation.  The conv decoder then
    predicts an ``R x R`` mask of the referred region.

    A large share of SRE samples have *no* referent (ground truth ``"No"``);
    those become all-zero masks, so the head naturally learns to emit an empty
    mask when the expression matches nothing.  Scored with the same per-pixel
    F1 (teochat ``Evaluator``) as the official ``evaluate_masks`` path.
    """

    def __init__(
        self,
        d_model: int,
        *,
        out_size: int,
        num_heads: int = 8,
        num_layers: int = 2,
        lambda_dice: float = 1.0,
        pos_weight: float = 5.0,
        hidden: int = 256,
        lambda_presence: float = 1.0,
        presence_threshold: float = 0.5,
        presence_pos_weight: float = 1.0,
        presence_source: str = "grid_stats",
        tversky_beta: float = 0.0,
    ):
        super().__init__()
        self.out_size = out_size
        self.lambda_dice = lambda_dice
        self.lambda_presence = lambda_presence
        self.presence_threshold = presence_threshold
        self.presence_source = presence_source
        self.tversky_beta = tversky_beta
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.register_buffer("presence_pos_weight", torch.tensor(float(presence_pos_weight)))
        # Spatial backbone (conv-on-grid, identical recipe to ChangeSegHead).
        self.grid_norm = nn.LayerNorm(d_model)
        self.proj = nn.Conv2d(d_model, hidden, kernel_size=1)
        # Referring-expression encoder: attention-pool the text-aware token
        # stream into one vector, then FiLM-modulate the conv features by it.
        # A heavier per-cell pixel<->text cross-attention was tried but
        # overfits this small (~4k example) split, so a single global FiLM
        # vector is kept -- it is the most reliable text-conditioning here.
        self.text_norm = nn.LayerNorm(d_model)
        self.text_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.text_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.film = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 2 * hidden)
        )
        body = []
        for _ in range(max(1, num_layers)):
            body += [
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            ]
        self.body = nn.Sequential(*body)
        self.out_conv = nn.Conv2d(hidden, 1, kernel_size=1)
        # Referent presence: ~50% of official SRE eval rows have empty GT
        # ("no such building").  Two signal sources are supported:
        #   - "grid_stats" (legacy): pooled text + global change magnitude.
        #     The change magnitude is identical for present/empty GT, so this
        #     collapses to "always present" and contributes no separation.
        #   - "mask": pooled text + statistics of the *predicted mask logit*
        #     (max, mean, positive-fraction).  The referent either produces a
        #     confident localized region or it does not -- so the mask's own
        #     peak is the right evidence for "does the referent exist".
        presence_extra = 3
        self.presence_head = nn.Sequential(
            nn.Linear(d_model + presence_extra, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def set_presence_pos_weight(self, weight: float) -> None:
        self.presence_pos_weight = torch.tensor(float(weight), device=self.pos_weight.device)

    def _pool_text(self, text_tokens: Tensor, token_mask: Tensor | None) -> Tensor:
        """Attention-pool the (text-aware) token stream to ``(B, d)``."""
        mem = self.text_norm(text_tokens)
        q = self.text_query.expand(mem.size(0), -1, -1)
        kpm = None if token_mask is None else ~token_mask.to(torch.bool)
        pooled, _ = self.text_attn(q, mem, mem, key_padding_mask=kpm, need_weights=False)
        return pooled.squeeze(1)

    @staticmethod
    def _grid_change_stats(grid: Tensor) -> Tensor:
        """``(B, G, G, d)`` -> ``(B, 3)`` magnitude summary (mean, max, top-fraction)."""
        mag = grid.norm(dim=-1)
        mean = mag.mean(dim=(1, 2))
        mx = mag.amax(dim=(1, 2))
        flat = mag.flatten(1)
        q = torch.quantile(flat, 0.9, dim=1)
        frac = (flat > q.unsqueeze(1)).float().mean(dim=1)
        return torch.stack([mean, mx, frac], dim=1)

    @staticmethod
    def _mask_stats(logits: Tensor) -> Tensor:
        """``(B, R, R)`` mask logits -> ``(B, 3)`` (max, mean, positive-fraction).

        Evidence for "does the referent exist": a present referent yields a
        confident localized peak; an absent one yields a flat low-prob field.
        """
        prob = logits.sigmoid()
        flat = prob.flatten(1)
        mx = flat.amax(dim=1)
        mean = flat.mean(dim=1)
        frac = (flat > 0.5).float().mean(dim=1)
        return torch.stack([mx, mean, frac], dim=1)

    def _presence_input(self, pooled: Tensor, grid: Tensor, logits: Tensor) -> Tensor:
        if self.presence_source == "mask":
            stats = self._mask_stats(logits)
        else:
            stats = self._grid_change_stats(grid)
        return torch.cat([pooled, stats.to(pooled.dtype)], dim=-1)

    def _predict(self, grid: Tensor, text_tokens: Tensor, token_mask: Tensor | None) -> Tensor:
        """``grid`` ``(B, G, G, d)`` + text -> ``(B, R, R)`` referred logits."""
        x = self.grid_norm(grid).permute(0, 3, 1, 2).contiguous()  # (B, d, G, G)
        x = self.proj(x)
        # FiLM-condition on the referring text at the native (coarse) grid.
        t = self._pool_text(text_tokens, token_mask)  # (B, d)
        gamma, beta = self.film(t).chunk(2, dim=-1)  # (B, hidden) each
        x = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        if x.shape[-1] != self.out_size:
            x = F.interpolate(
                x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False
            )
        x = self.body(x)
        return self.out_conv(x).squeeze(1)  # (B, R, R)

    def forward(
        self,
        grid: Tensor,
        *,
        text_tokens: Tensor | None = None,
        target_mask: Tensor | None = None,
        token_mask: Tensor | None = None,
        **_ignore,
    ) -> dict:
        if text_tokens is None:
            text_tokens = grid.flatten(1, 2)  # fall back to the grid as text-free memory
        pooled = self._pool_text(text_tokens, token_mask)  # (B, d)
        logits = self._predict(grid, text_tokens, token_mask)
        presence_in = self._presence_input(pooled, grid, logits)
        presence_logit = self.presence_head(presence_in).squeeze(-1)  # (B,)
        out: dict = {"mask_logits": logits, "presence_logit": presence_logit}
        if target_mask is None:
            return out
        target = target_mask.to(logits.dtype)
        present_f = (target.flatten(1).sum(dim=1) > 0).to(logits.dtype)
        pw = self.presence_pos_weight.to(logits.dtype)
        presence = F.binary_cross_entropy_with_logits(presence_logit, present_f, pos_weight=pw)
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight.to(logits.dtype)
        )
        prob = logits.sigmoid()
        dims = (1, 2)
        if self.tversky_beta > 0:
            # Tversky: TP / (TP + alpha*FN + beta*FP).  beta>0.5 penalises
            # false-positives more, curbing the 2x over-prediction that forces
            # a high binarization threshold at eval.
            beta = self.tversky_beta
            alpha = 1.0 - beta
            tp = (prob * target).sum(dim=dims)
            fp = (prob * (1.0 - target)).sum(dim=dims)
            fn = ((1.0 - prob) * target).sum(dim=dims)
            tversky = 1.0 - ((tp + 1.0) / (tp + alpha * fn + beta * fp + 1.0)).mean()
            region = tversky
            out["loss_dice"] = tversky.detach()
        else:
            inter = (prob * target).sum(dim=dims)
            denom = prob.sum(dim=dims) + target.sum(dim=dims)
            dice = 1.0 - ((2 * inter + 1.0) / (denom + 1.0)).mean()
            region = dice
            out["loss_dice"] = dice.detach()
        out["loss"] = bce + self.lambda_dice * region + self.lambda_presence * presence
        out["loss_bce"] = bce.detach()
        out["loss_presence"] = presence.detach()
        return out

    @torch.inference_mode()
    def decode(
        self,
        grid: Tensor,
        *,
        text_tokens: Tensor | None = None,
        eval_size: int = 256,
        threshold: float = 0.5,
        token_mask: Tensor | None = None,
        **_ignore,
    ) -> Tensor:
        if text_tokens is None:
            text_tokens = grid.flatten(1, 2)
        pooled = self._pool_text(text_tokens, token_mask)
        logits = self._predict(grid, text_tokens, token_mask)
        presence_in = self._presence_input(pooled, grid, logits)
        presence = self.presence_head(presence_in).squeeze(-1).sigmoid()  # (B,)
        up = F.interpolate(
            logits.unsqueeze(1), size=(eval_size, eval_size), mode="bilinear", align_corners=False
        ).squeeze(1)
        mask = (up.sigmoid() > threshold).to(torch.uint8)
        # Gate: drop the whole mask when no referent is predicted present.
        gate = (presence > self.presence_threshold).to(torch.uint8)
        return mask * gate[:, None, None]


def build_head(cfg: HeadConfig, *, d_model: int) -> nn.Module:
    if cfg.kind == "classifier":
        return ClassifierHead(
            d_model,
            num_classes=cfg.num_classes,
            pool=cfg.pool,
            focal_gamma=float(getattr(cfg, "cls_focal_gamma", 0.0)),
        )
    if cfg.kind == "bbox_grid":
        return BBoxGridHead(
            d_model,
            grid_side=cfg.grid_side,
            lambda_box=cfg.bbox_lambda_box,
            lambda_obj=cfg.bbox_lambda_obj,
            focal_alpha=cfg.bbox_focal_alpha,
            focal_gamma=cfg.bbox_focal_gamma,
        )
    if cfg.kind == "text_decoder":
        return TextDecoderHead(
            d_model,
            vocab_size=cfg.text_vocab_size,
            max_len=cfg.text_max_len,
            num_layers=cfg.text_num_layers,
            num_heads=cfg.text_num_heads,
            dropout=cfg.text_dropout,
        )
    if cfg.kind == "change_seg":
        return ChangeSegHead(
            d_model,
            out_size=cfg.seg_out_size,
            num_heads=cfg.seg_heads,
            num_layers=cfg.seg_num_layers,
            lambda_dice=cfg.seg_lambda_dice,
            pos_weight=cfg.seg_pos_weight,
        )
    if cfg.kind == "referring_seg":
        return ReferringSegHead(
            d_model,
            out_size=cfg.seg_out_size,
            num_heads=cfg.seg_heads,
            num_layers=cfg.seg_num_layers,
            lambda_dice=cfg.seg_lambda_dice,
            pos_weight=cfg.seg_pos_weight,
            lambda_presence=cfg.seg_lambda_presence,
            presence_threshold=cfg.seg_presence_threshold,
            presence_pos_weight=cfg.seg_presence_pos_weight,
            presence_source=cfg.seg_presence_source,
            tversky_beta=cfg.seg_tversky_beta,
        )
    if cfg.kind == "gemma_qa":
        return GemmaQADecoderHead(
            d_model,
            gemma_model_dir=getattr(cfg, "gemma_model_dir", "models/google/gemma-4-E4B-it"),
            n_adapter_layers=getattr(cfg, "n_adapter_layers", 4),
            adapter_heads=getattr(cfg, "adapter_heads", 8),
            max_gen_len=getattr(cfg, "max_gen_len", 64),
        )
    raise ValueError(f"unknown head kind: {cfg.kind!r}")


class GemmaQADecoderHead(nn.Module):
    """QA head that decodes through frozen Gemma4 language model layers.

    Feeds the Mamba trunk output into Gemma4's frozen text decoder via
    learnable cross-attention adapters, then uses the frozen LM head to
    produce token probabilities.  Only the projection and cross-attention
    adapter parameters are trained (~5-10M); Gemma4 itself stays frozen.

    This lets the QA head leverage Gemma4's pretrained language generation
    ability while being conditioned on spatially-aligned visual features.
    """

    def __init__(
        self,
        d_model: int,
        *,
        gemma_model_dir: str,
        n_adapter_layers: int = 4,
        adapter_heads: int = 8,
        max_gen_len: int = 64,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_gen_len = max_gen_len
        self.dtype = dtype

        model = AutoModelForImageTextToText.from_pretrained(gemma_model_dir, dtype=dtype)
        for p in model.parameters():
            p.requires_grad = False
        self.gemma = model
        self.processor = AutoProcessor.from_pretrained(gemma_model_dir)
        self.tokenizer = self.processor.tokenizer

        d_gemma = model.config.text_config.hidden_size  # 2560
        self.proj_in = nn.Linear(d_model, d_gemma)

        self.cross_attn_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn": nn.MultiheadAttention(
                            d_gemma, adapter_heads, dropout=0.1, batch_first=True
                        ),
                        "norm1": nn.LayerNorm(d_gemma),
                        "norm2": nn.LayerNorm(d_gemma),
                        "ffn": nn.Sequential(
                            nn.Linear(d_gemma, d_gemma * 2),
                            nn.GELU(),
                            nn.Linear(d_gemma * 2, d_gemma),
                            nn.Dropout(0.1),
                        ),
                    }
                )
                for _ in range(n_adapter_layers)
            ]
        )

        self.out_proj = nn.Linear(d_gemma, model.config.text_config.vocab_size, bias=False)
        with torch.no_grad():
            lm_weight = model.language_model.lm_head.weight.data
            if self.out_proj.weight.shape == lm_weight.shape:
                self.out_proj.weight.copy_(lm_weight)
        self.out_proj.requires_grad_(True)

    def _cross_attend(self, text_hidden: Tensor, visual_ctx: Tensor) -> Tensor:
        h = text_hidden
        for layer in self.cross_attn_layers:
            residual = h
            h_norm = layer["norm1"](h)
            h_ca, _ = layer["cross_attn"](h_norm, visual_ctx, visual_ctx)
            h = residual + h_ca
            residual = h
            h = residual + layer["ffn"](layer["norm2"](h))
        return h

    def forward(
        self,
        tokens: Tensor,
        *,
        text_ids: Tensor | None = None,
        token_mask: Tensor | None = None,
    ) -> dict[str, Any]:
        tokens.size(0)
        visual_ctx = self.proj_in(tokens.to(self.proj_in.weight.dtype))

        if text_ids is not None:
            target_ids = text_ids[:, 1:]
            input_ids = text_ids[:, :-1]
            with torch.no_grad():
                emb = self.gemma.language_model.model.embed_tokens(input_ids)
            h = self._cross_attend(emb.to(visual_ctx.dtype), visual_ctx)
            logits = self.out_proj(h)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1).long(),
                ignore_index=self.tokenizer.pad_token_id or 0,
            )
            preds = logits.argmax(dim=-1)
            pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            return {"loss": loss, "predictions": pred_texts}

        return self.generate(tokens, token_mask=token_mask)

    @torch.inference_mode()
    def generate(
        self,
        tokens: Tensor,
        *,
        token_mask: Tensor | None = None,
        max_len: int | None = None,
    ) -> list[str]:
        max_len = max_len or self.max_gen_len
        B = tokens.size(0)
        visual_ctx = self.proj_in(tokens.to(self.proj_in.weight.dtype))

        bos_id = self.tokenizer.bos_token_id or 2
        input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=tokens.device)
        for _ in range(max_len):
            with torch.no_grad():
                emb = self.gemma.language_model.model.embed_tokens(input_ids)
            h = self._cross_attend(emb.to(visual_ctx.dtype), visual_ctx)
            logits = self.out_proj(h[:, -1:, :])
            next_id = logits.argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if (next_id == self.tokenizer.eos_token_id).all():
                break

        texts = self.tokenizer.batch_decode(input_ids[:, 1:], skip_special_tokens=True)
        return texts


class TaskAdapter(nn.Module):
    """Lightweight per-task bottleneck adapter between the shared trunk and head.

    Transforms ``(B, L, d_model) -> (B, L, d_model)`` via a 2-layer MLP with
    a narrow bottleneck (default ``d_model // 4``).  Each task gets its own
    adapter, so task-specific gradients flow through the adapter without
    conflicting inside the shared trunk.  This prevents negative transfer
    observed in naive multi-task trunk sharing while keeping the trunk fully
    shared.

    Residual + LayerNorm ensures the adapter starts near-identity, stabilising
    early training.
    """

    def __init__(self, d_model: int, bottleneck: int = 0, dropout: float = 0.1):
        super().__init__()
        bn = bottleneck if bottleneck > 0 else d_model // 4
        self.down = nn.Linear(d_model, bn)
        self.act = nn.GELU()
        self.up = nn.Linear(bn, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.drop(self.up(self.act(self.down(x)))))


__all__ = [
    "BBoxGridHead",
    "BOS_ID",
    "ChangeSegHead",
    "ClassifierHead",
    "EOS_ID",
    "GemmaQADecoderHead",
    "PAD_ID",
    "ReferringSegHead",
    "TaskAdapter",
    "TEXT_VOCAB",
    "TEXT_VOCAB_SIZE",
    "TextDecoderHead",
    "build_head",
    "decode_text",
    "encode_text",
]
