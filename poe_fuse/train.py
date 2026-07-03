"""Train a PoEFuse model on one of the four target tasks.

Tasks:

* ``xbd_dmg_cls``    -- 5-way classifier head; the marquee metric is the
                        upstream inverse-prevalence weighted per-pixel F1
                        (``change_detection_classification_f1`` returned
                        by :func:`teochat_eval.detection_metrics`).
* ``s2_det``         -- grid-anchor bbox head; per-pixel F1 between the
                        rasterized prediction bboxes and the GT polygons.
* ``xbd_loc``        -- same per-pixel F1 metric as ``s2_det``.
* ``s2looking_sre``  -- char-level text decoder; per-pixel F1 from bbox
                        strings parsed out of the generated text.

All eval-side scoring is delegated to the vendored official TEOChat
evaluator (:mod:`teochat_eval`), which is a byte-for-byte copy of
``videollava/eval/{classification,detection}.py``.  No metric is
re-implemented in this file.

The head is selected by ``head.kind`` in the YAML config; the dataset's
``label_kind`` must agree.

Usage::

    uv run src/train.py --config configs/poe_fuse/<task>.yaml

Early stopping is on by default: if ``eval_loss`` does not improve for
``train.early_stop_patience`` consecutive epochs (default 5) the loop
breaks and the best epoch's weights are restored before the final eval
+ checkpoint save.  Set ``train.early_stop_patience: 0`` in the YAML to
disable.

Set ``WANDB_MODE=disabled`` to skip wandb logging entirely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from poe_fuse import (
    CLS_VOCABS,
    DEFAULT_IMAGE_BASE_CANDIDATES,
    XBD_DAMAGE_LABELS,
    CachedTrioDataset,
    ChangeSegHead,
    FusionConfig,
    HeadConfig,
    ImagePairDataset,
    Mamba3StackConfig,
    MixerConfig,
    PoEFuse,
    PoEFuseConfig,
    ReferringSegHead,
    TrioBackboneConfig,
    cached_collate,
    features_to_device,
    format_bbox_list,
    sanitize_bbox_response,
)
from poe_fuse.dataloader import DatasetTaskSpec, spec_by_key
from poe_fuse.metrics import classification_metrics, detection_metrics
from poe_fuse.metrics.detection import Evaluator as _PixelEvaluator

# ---------------------------------------------------------------- config


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"yaml root must be a mapping: {path}")
    return raw


def _build_poe_fuse_config(raw: dict[str, Any]) -> PoEFuseConfig:
    backbone = TrioBackboneConfig(**(raw.get("backbone") or {}))
    mamba3 = Mamba3StackConfig(**(raw.get("mamba3") or {}))
    head = HeadConfig(**(raw.get("head") or {}))
    fusion = FusionConfig(**(raw.get("fusion") or {}))
    mixer = MixerConfig(**(raw.get("mixer") or {}))
    return PoEFuseConfig(backbone=backbone, mamba3=mamba3, head=head, fusion=fusion, mixer=mixer)


def _label_kind(head_kind: str) -> str:
    return {
        "classifier": "cls",
        "bbox_grid": "bbox",
        "text_decoder": "text",
        # The dense change-segmentation head reuses the bbox cache (it carries
        # the GT change polygons) and rasterises masks from them on the fly.
        "change_seg": "bbox",
        # The referring-seg head runs on the SRE cache (label_kind "text") and
        # rasterises the GT referent polygon (empty -> all-zero mask).
        "referring_seg": "text",
        "gemma_qa": "text",
    }[head_kind]


def _rasterize_polys(polys: list[str], size: int) -> np.ndarray:
    """Rasterise a list of WKT polygons (256-px coords) into a 0/1 mask."""
    from PIL import Image, ImageDraw
    from shapely import wkt as _wkt

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    for p in polys or []:
        if not p or "POLYGON" not in p:
            continue
        try:
            geom = _wkt.loads(p)
        except Exception:
            continue
        try:
            draw.polygon(list(geom.exterior.coords), outline=1, fill=1)
        except Exception:
            continue
    return np.asarray(img, dtype=np.float32)


def _rasterize_boxes(boxes: list[list[float]], size: int) -> np.ndarray:
    """Rasterise [x1,y1,x2,y2] boxes (0-100 normalised) into a 0/1 mask."""
    from PIL import Image, ImageDraw

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    s = size / 100.0
    for b in boxes or []:
        try:
            x1, y1, x2, y2 = (float(b[0]) * s, float(b[1]) * s, float(b[2]) * s, float(b[3]) * s)
        except Exception:
            continue
        draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], outline=1, fill=1)
    return np.asarray(img, dtype=np.float32)


def _parse_boxes_from_text(text: str) -> list[list[float]]:
    """Parse ``[x1, y1, x2, y2]`` boxes (0-100) out of a GT/response string."""
    boxes: list[list[float]] = []
    for grp in re.findall(r"\[(.*?)\]", text or ""):
        try:
            vals = [float(x) for x in grp.split(",")]
        except ValueError:
            continue
        if len(vals) == 4:
            boxes.append(vals)
    return boxes


def _filter_sre_eval(eval_ds: Any, train_cfg: dict[str, Any]) -> Any:
    """Restrict the S2Looking SRE eval split to the official metric subset.

    The ``S2Looking_SRE_QA.json`` eval file mixes two tasks:
    ``spatial_referring_expression`` (bbox referents) and ``question_answering``
    (Yes/No etc.).  TEOChat's ``detection_metrics`` groups by ``task`` and
    reports ``spatial_referring_expression_f1`` over the SRE rows *only*; the QA
    rows are scored separately as accuracy and are **not** part of the SRE F1.
    Evaluating pixel-F1 over the full mixed set (as we did initially) is simply
    the wrong denominator.  This filters the eval dataset to the SRE rows by
    matching ``sample_id`` against the source JSON's ``task`` field.
    """
    if train_cfg.get("task") != "spatial_referring_expression":
        return eval_ds
    src = train_cfg.get("src_json")
    if not src or not Path(str(src)).is_file():
        print("[sre-eval-filter] src_json missing; eval NOT filtered", flush=True)
        return eval_ds
    rows = json.loads(Path(str(src)).read_text())
    sre_ids = {str(r.get("id")) for r in rows if r.get("task") == "spatial_referring_expression"}
    base = eval_ds.dataset if isinstance(eval_ds, Subset) else eval_ds
    keep = [i for i in range(len(base)) if str(base[i].get("sample_id")) in sre_ids]
    if not keep:
        print("[sre-eval-filter] no SRE rows matched; eval NOT filtered", flush=True)
        return eval_ds
    print(
        f"[sre-eval-filter] kept {len(keep)}/{len(base)} "
        f"spatial_referring_expression rows (dropped QA rows)",
        flush=True,
    )
    return Subset(base, keep)


def _has_referent(sample: dict[str, Any]) -> bool:
    """True when the sample carries a non-empty referent region."""
    if sample.get("polygons"):
        return True
    if sample.get("bboxes"):
        return True
    return bool(_parse_boxes_from_text(str(sample.get("ground_truth") or "")))


def _sre_referent_split(dataset: Any) -> tuple[list[int], list[int]]:
    """Partition dataset indices into referent-present vs empty-GT."""
    pos: list[int] = []
    neg: list[int] = []
    for i in range(len(dataset)):
        if _has_referent(dataset[i]):
            pos.append(i)
        else:
            neg.append(i)
    return pos, neg


class _SREBalancedBatchSampler:
    """Yield balanced batches (half referent / half empty) for SRE training."""

    def __init__(self, pos_idx: list[int], neg_idx: list[int], batch_size: int):
        if not pos_idx or not neg_idx:
            raise ValueError("balanced SRE sampler needs both referent and empty indices")
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.batch_size = batch_size
        self.n_pos = batch_size // 2
        self.n_neg = batch_size - self.n_pos

    def __len__(self) -> int:
        steps = max(
            (len(self.pos_idx) + self.n_pos - 1) // self.n_pos,
            (len(self.neg_idx) + self.n_neg - 1) // self.n_neg,
        )
        return steps

    def __iter__(self):
        import random

        pos = self.pos_idx.copy()
        neg = self.neg_idx.copy()
        random.shuffle(pos)
        random.shuffle(neg)
        pi = ni = 0
        for _ in range(len(self)):
            batch: list[int] = []
            for _ in range(self.n_pos):
                batch.append(pos[pi % len(pos)])
                pi += 1
            for _ in range(self.n_neg):
                batch.append(neg[ni % len(neg)])
                ni += 1
            yield batch


def _seg_targets(batch: dict[str, Any], out_size: int, device: torch.device) -> torch.Tensor:
    """Build ``(B, out_size, out_size)`` 0/1 mask targets for a batch.

    Target geometry is taken from the first available source, in order:

    1. GT **polygons** (change-seg eval splits, SRE eval referents);
    2. GT **bboxes** (s2_det / xbd_loc train carry boxes, not WKT polygons);
    3. boxes **parsed from the ``ground_truth`` string** (SRE train/eval encode
       the referent box as text like ``[83, 12, 100, 35].``; ``"No"`` / prose
       answers yield an empty mask).

    Sources are rasterised at 256 px then max-pooled to ``out_size`` so small
    footprints survive.
    """
    polys = batch.get("polygons") or []
    boxes = batch.get("bbox_targets") or []
    gts = batch.get("ground_truths") or []
    B = len(batch["sample_ids"])
    masks = []
    for i in range(B):
        p = polys[i] if i < len(polys) else []
        b = boxes[i] if i < len(boxes) else []
        if p:
            m = _rasterize_polys(p, 256)
        elif b:
            m = _rasterize_boxes(b, 256)
        else:
            gt = gts[i] if i < len(gts) else ""
            m = _rasterize_boxes(_parse_boxes_from_text(gt), 256)
        masks.append(torch.from_numpy(m))
    full = torch.stack(masks, dim=0).unsqueeze(1)  # (B,1,256,256)
    pooled = torch.nn.functional.max_pool2d(full, kernel_size=256 // out_size)
    return (pooled.squeeze(1) > 0).float().to(device)


def _maybe_init_wandb(
    train_cfg: dict[str, Any],
    cfg: PoEFuseConfig,
    *,
    config_path: Path,
) -> tuple[Any, str]:
    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "disabled":
        return None, ""
    try:
        import wandb
    except ImportError:
        return None, ""
    project = os.environ.get("WANDB_PROJECT") or train_cfg.get("wandb_project")
    if not project:
        return None, ""
    entity = os.environ.get("WANDB_ENTITY") or train_cfg.get("wandb_entity")
    name = (
        train_cfg.get("wandb_run_name")
        or train_cfg.get("model_name")
        or train_cfg.get("dataset_key")
    )
    group = train_cfg.get("wandb_group") or "poe_fuse_pair_diff"
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        mode=mode,
        config={
            "config_path": str(config_path),
            "dataset_key": train_cfg.get("dataset_key"),
            "task": train_cfg.get("task"),
            "head_kind": cfg.head.kind,
            "backbone": asdict(cfg.backbone),
            "mamba3": asdict(cfg.mamba3),
            "head": asdict(cfg.head),
            "train": train_cfg,
        },
    )
    return run, getattr(run, "url", "") or ""


# ---------------------------------------------------------------- utility


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = {b["label_kind"] for b in batch}
    if len(kinds) != 1:
        raise RuntimeError(f"mixed label_kinds in batch: {kinds}")
    kind = kinds.pop()

    # ``images`` is (T_i, 3, H, W) and ``T_i`` is variable per sample;
    # zero-pad each sample to the batch max ``T_max`` along the T axis and
    # build an ``image_mask`` so downstream attention / pooling can ignore
    # the padded slots.
    per_image = [b["images"] for b in batch]
    T_max = max(int(img.size(0)) for img in per_image)
    B = len(per_image)
    sample0 = per_image[0]
    images = sample0.new_zeros(B, T_max, *sample0.shape[1:])
    image_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, img in enumerate(per_image):
        T_i = int(img.size(0))
        images[i, :T_i] = img
        image_mask[i, :T_i] = True

    out: dict[str, Any] = {
        "images": images,
        "image_mask": image_mask,
        "sample_ids": [b["sample_id"] for b in batch],
        "ground_truths": [b["ground_truth"] for b in batch],
        "questions": [str(b.get("question") or "") for b in batch],
        "polygons": [list(b.get("polygons") or []) for b in batch],
        "metas": [b["meta"] for b in batch],
        "label_kind": kind,
    }
    if kind == "cls":
        out["labels"] = torch.tensor([int(b["label_id"]) for b in batch], dtype=torch.long)
    elif kind == "bbox":
        out["bbox_targets"] = [list(b["bboxes"]) for b in batch]
    elif kind == "text":
        out["text_ids"] = torch.tensor([list(b["text_ids"]) for b in batch], dtype=torch.long)
    else:
        raise ValueError(f"unsupported label_kind: {kind}")
    return out


# ---------------------------------------------------------------- official eval


def _resolve_task_spec(train_cfg: dict[str, Any], head_kind: str) -> DatasetTaskSpec:
    """Look up the :class:`DatasetTaskSpec` describing this run.

    We need the spec at eval time to know:

    * which ``record['task']`` tag to stamp on each prediction record;
    * which ``dataset_name`` to pass to the upstream
      :func:`teochat_eval.detection_metrics`; and
    * which key in its return dict to treat as the marquee score.
    """
    key = train_cfg.get("dataset_key") or train_cfg.get("task")
    if not key:
        raise ValueError(
            "train config must specify `dataset_key` (or `task`) so we can "
            "route the eval through the official TEOChat evaluator"
        )
    spec = spec_by_key(str(key))
    # ``change_seg`` is a drop-in alternative head for the bbox detection tasks
    # (it predicts a dense change mask instead of grid boxes but is scored by
    # the same per-pixel F1), so accept it wherever ``bbox_grid`` is expected.
    compatible = {spec.head_kind}
    if spec.head_kind == "bbox_grid":
        compatible.add("change_seg")
    if spec.head_kind == "text_decoder":
        compatible.add("referring_seg")
    if spec.head_kind == "classifier":
        compatible.add("gemma_qa")
        compatible.add("text_decoder")
    if head_kind not in compatible:
        raise ValueError(
            f"dataset_key={key!r} expects head_kind={spec.head_kind!r} but the "
            f"YAML config wires head_kind={head_kind!r}"
        )
    return spec


def _score_records(records: list[dict[str, Any]], spec: DatasetTaskSpec) -> dict[str, float]:
    """Call the vendored official ``detection_metrics`` on ``records``.

    All entries are tagged with ``task=spec.task``; the upstream function
    groups them by task internally.  The marquee metric is exposed under
    its native key (e.g. ``change_detection_classification_f1``) so the
    downstream leaderboard plumbing can pick it up with the existing
    ``spec.metric`` accessor.
    """
    if not records:
        return {spec.metric: 0.0, "n_predictions": 0.0}
    if getattr(spec, "eval_scorer", "detection") == "classification":
        # Exact-match accuracy (teochat classification_metrics), e.g. fMoW
        # scene classification -> ``{task}_accuracy``.
        raw = classification_metrics(records)
        return {k: float(v) for k, v in raw.items()}
    raw = detection_metrics(records, spec.eval_dataset_name)
    return {k: float(v) for k, v in raw.items()}


# ---------------------------------------------------------------- loops


def _resolve_sam3_prompts(spec: DatasetTaskSpec, questions: list[str]) -> list[str]:
    """Pick the SAM 3 text prompt per sample.

    Detection tasks use the spec's short noun phrase (``"building"`` etc.);
    the SRE task has an empty default so we fall back to the per-sample
    referring expression itself.
    """
    default = spec.sam3_default_prompt or ""
    if default:
        return [default] * len(questions)
    return [q or "object" for q in questions]


def _codec_tokens(
    model: PoEFuse,
    batch: dict[str, Any],
    *,
    device: torch.device,
    spec: DatasetTaskSpec,
    from_cache: bool,
) -> dict[str, Any]:
    """Produce ``{tokens, token_mask}`` from either raw images or cached feats.

    * ``from_cache=False``: run the full codec (frozen backbones + trainable
      mix) over ``batch["images"]``.
    * ``from_cache=True``: skip the backbones and run only ``codec.mix`` over
      the pre-computed per-branch features carried by the cached batch.
    """
    if from_cache:
        raw = features_to_device(batch, device)
        return model.codec.mix(raw)
    images = batch["images"].to(device, non_blocking=True)
    image_mask = batch.get("image_mask")
    if image_mask is not None:
        image_mask = image_mask.to(device, non_blocking=True)
    questions = batch["questions"]
    sam3_prompts = _resolve_sam3_prompts(spec, questions)
    return model.codec(images, questions, sam3_prompts, image_mask=image_mask)


def _train_forward(
    model: PoEFuse,
    batch: dict[str, Any],
    *,
    device: torch.device,
    spec: DatasetTaskSpec,
    from_cache: bool,
) -> dict[str, Any]:
    codec_out = _codec_tokens(model, batch, device=device, spec=spec, from_cache=from_cache)
    if model.adapter is not None:
        codec_out = {**codec_out, "tokens": model.adapter(codec_out["tokens"])}
    if isinstance(model.head, ChangeSegHead):
        target = _seg_targets(batch, model.head.out_size, device)
        return model.head(codec_out["change_grid"], target_mask=target)
    if isinstance(model.head, ReferringSegHead):
        target = _seg_targets(batch, model.head.out_size, device)
        return model.head(
            codec_out["change_grid"],
            text_tokens=codec_out["tokens"],
            target_mask=target,
            token_mask=codec_out["token_mask"],
        )
    kind = batch["label_kind"]
    if kind == "cls":
        labels = batch["labels"].to(device, non_blocking=True)
        return model._dispatch_head(codec_out, labels=labels, bbox_targets=None, text_ids=None)
    if kind == "bbox":
        return model._dispatch_head(
            codec_out, labels=None, bbox_targets=batch["bbox_targets"], text_ids=None
        )
    if kind == "text":
        text_ids = batch["text_ids"].to(device, non_blocking=True)
        return model._dispatch_head(codec_out, labels=None, bbox_targets=None, text_ids=text_ids)
    raise ValueError(kind)


def _prediction_text_cls(pred_id: int, vocab: tuple[str, ...] = XBD_DAMAGE_LABELS) -> str:
    """Map a classifier head's predicted class id to its text label."""
    if 0 <= pred_id < len(vocab):
        return vocab[pred_id]
    return "unknown"


def _run_eval_seg(
    model: PoEFuse,
    loader: DataLoader,
    device: torch.device,
    *,
    cfg: PoEFuseConfig,
    spec: DatasetTaskSpec,
    from_cache: bool = False,
) -> dict[str, Any]:
    """Eval for the dense change-seg head: per-pixel F1 of predicted vs GT mask.

    Uses the same teochat ``Evaluator`` (num_class=2, change=class 1) and the
    same 256-px GT rasterisation as the official ``evaluate_masks`` path, so
    the reported ``f1`` is directly comparable to the bbox proxy -- only the
    prediction is a dense mask instead of grid boxes.
    """
    out_size = model.head.out_size
    is_referring = isinstance(model.head, ReferringSegHead)
    evaluator = _PixelEvaluator(num_class=2)
    total_loss = 0.0
    total_n = 0
    n_pred_pos = 0
    with torch.inference_mode():
        for batch in loader:
            batch_polys = batch.get("polygons") or [[] for _ in batch["sample_ids"]]
            codec_out = _codec_tokens(model, batch, device=device, spec=spec, from_cache=from_cache)
            if model.adapter is not None:
                codec_out = {**codec_out, "tokens": model.adapter(codec_out["tokens"])}
            token_mask = codec_out["token_mask"]
            head_in = codec_out["change_grid"]
            text_kw = {"text_tokens": codec_out["tokens"]} if is_referring else {}
            target = _seg_targets(batch, out_size, device)
            head_out = model.head(head_in, target_mask=target, token_mask=token_mask, **text_kw)
            total_loss += float(head_out["loss"].detach()) * len(batch_polys)
            total_n += len(batch_polys)
            pred = (
                model.head.decode(
                    head_in,
                    eval_size=256,
                    threshold=cfg.head.seg_threshold,
                    token_mask=token_mask,
                    **text_kw,
                )
                .cpu()
                .numpy()
            )
            n_pred_pos += int(pred.sum())
            for b, polys in enumerate(batch_polys):
                gt = (_rasterize_polys(polys, 256) > 0).astype("uint8")
                evaluator.add_batch(gt, pred[b])

    f1_arr = np.asarray(evaluator.Pixel_F1_score(), dtype=float)
    f1 = float(np.nanmean(f1_arr)) if f1_arr.size else float("nan")
    iou_arr = np.asarray(evaluator.Intersection_over_Union(), dtype=float)
    metrics = {
        spec.metric: f1,
        "eval_loss": total_loss / max(1, total_n),
        "change_iou": float(np.nanmean(iou_arr)) if iou_arr.size else float("nan"),
        "n_predictions": float(total_n),
        "pred_pos_px_per_sample": float(n_pred_pos) / max(1, total_n),
    }
    return {"metrics": metrics, "predictions": []}


def _run_eval(
    model: PoEFuse,
    loader: DataLoader,
    device: torch.device,
    *,
    cfg: PoEFuseConfig,
    spec: DatasetTaskSpec,
    from_cache: bool = False,
) -> dict[str, Any]:
    """Generate predictions, score them with the vendored official eval.

    The function builds one record per sample in the format consumed by
    :func:`teochat_eval.detection_metrics`::

        {
            "task":          spec.task,
            "response":      <model output text>,
            "ground_truth":  <reference text>,
            "polygon":       [<wkt str>, ...],
        }

    and calls the upstream evaluator on the full list.  ``eval_loss`` is
    still computed from the model's training-side loss (the only number
    that is *not* coming from the upstream evaluator -- it has no
    equivalent there).
    """
    model.eval()

    # Dense segmentation heads (change or referring): predicted mask vs
    # rasterised GT mask, scored with the *same* per-pixel F1 (teochat
    # ``Evaluator``) the bbox/text paths use.
    if isinstance(model.head, (ChangeSegHead, ReferringSegHead)):
        return _run_eval_seg(model, loader, device, cfg=cfg, spec=spec, from_cache=from_cache)

    kind = _label_kind(cfg.head.kind)
    rows_out: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    total_loss = 0.0
    total_n = 0

    with torch.inference_mode():
        for batch in loader:
            batch_polys = batch.get("polygons") or [[] for _ in batch["sample_ids"]]
            n_in_batch = len(batch["sample_ids"])
            codec_out = _codec_tokens(model, batch, device=device, spec=spec, from_cache=from_cache)
            if model.adapter is not None:
                codec_out = {**codec_out, "tokens": model.adapter(codec_out["tokens"])}
            tokens = codec_out["tokens"]
            token_mask = codec_out["token_mask"]
            if kind == "cls":
                labels = batch["labels"].to(device, non_blocking=True)
                head_out = model.head(tokens, labels=labels, token_mask=token_mask)
                total_loss += float(head_out["loss"].detach()) * labels.size(0)
                total_n += labels.size(0)
                preds = head_out["predictions"].tolist()
                cls_vocab = CLS_VOCABS.get(
                    getattr(spec, "cls_vocab", "xbd_damage"), XBD_DAMAGE_LABELS
                )
                for sid, gt_text, gt_id, pr, polys in zip(
                    batch["sample_ids"],
                    batch["ground_truths"],
                    labels.tolist(),
                    preds,
                    batch_polys,
                ):
                    pred_text = _prediction_text_cls(int(pr), cls_vocab)
                    rows_out.append(
                        {
                            "sample_id": sid,
                            "ground_truth": gt_text,
                            "ground_truth_id": int(gt_id),
                            "prediction_id": int(pr),
                            "prediction": pred_text,
                            "polygons": list(polys),
                        }
                    )
                    records.append(
                        {
                            "task": spec.task,
                            "response": pred_text,
                            "ground_truth": str(gt_text),
                            "polygon": list(polys),
                        }
                    )
            elif kind == "bbox":
                targets = batch["bbox_targets"]
                head_out = model.head(
                    tokens,
                    bbox_targets=[[tuple(b) for b in t] for t in targets],
                    token_mask=token_mask,
                )
                decoded = model.head.decode(
                    tokens,
                    threshold=cfg.head.bbox_obj_threshold,
                    token_mask=token_mask,
                )
                total_loss += float(head_out["loss"].detach()) * n_in_batch
                total_n += n_in_batch
                for sid, gt_text, gt_b, pr_b, polys in zip(
                    batch["sample_ids"],
                    batch["ground_truths"],
                    targets,
                    decoded,
                    batch_polys,
                ):
                    pred_text = format_bbox_list(pr_b)
                    rows_out.append(
                        {
                            "sample_id": sid,
                            "ground_truth": gt_text,
                            "ground_truth_bboxes": list(gt_b),
                            "prediction_bboxes": list(pr_b),
                            "prediction": pred_text,
                            "polygons": list(polys),
                        }
                    )
                    records.append(
                        {
                            "task": spec.task,
                            "response": pred_text,
                            "ground_truth": str(gt_text),
                            "polygon": list(polys),
                        }
                    )
            elif kind == "text":
                text_ids = batch["text_ids"].to(device, non_blocking=True)
                head_out = model.head(tokens, text_ids=text_ids, token_mask=token_mask)
                generated = model.head.generate(tokens, token_mask=token_mask)
                total_loss += float(head_out["loss"].detach()) * text_ids.size(0)
                total_n += text_ids.size(0)
                for sid, gt_text, prediction, polys in zip(
                    batch["sample_ids"],
                    batch["ground_truths"],
                    generated,
                    batch_polys,
                ):
                    pred_text = str(prediction)
                    rows_out.append(
                        {
                            "sample_id": sid,
                            "ground_truth": gt_text,
                            "prediction": pred_text,
                            "polygons": list(polys),
                        }
                    )
                    records.append(
                        {
                            "task": spec.task,
                            "response": sanitize_bbox_response(pred_text),
                            "ground_truth": str(gt_text),
                            "polygon": list(polys),
                        }
                    )
            else:
                raise ValueError(kind)

    metrics = _score_records(records, spec)
    metrics["eval_loss"] = total_loss / max(1, total_n)
    metrics["n_predictions"] = float(len(rows_out))
    return {"metrics": metrics, "predictions": rows_out}


def _class_weights(dataset: Any, *, num_classes: int) -> torch.Tensor | None:
    """Inverse-frequency CE weights from the training label distribution.

    Works for both :class:`ImagePairDataset` (reads ``rows[i]['label_id']``)
    and :class:`CachedTrioDataset` (reads the manifest ``label_ids``).  Absent
    classes get weight 0 so they don't dominate the normalisation.  Returns
    ``None`` if no label ids are available.
    """
    label_ids: list[int] | None = None
    cached = getattr(dataset, "label_ids", None)
    if cached is not None:
        label_ids = list(cached)
    elif hasattr(dataset, "rows"):
        label_ids = [int(r["label_id"]) for r in dataset.rows if "label_id" in r]
    if not label_ids:
        return None
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for lid in label_ids:
        if 0 <= lid < num_classes:
            counts[lid] += 1
    total = float(counts.sum().item())
    if total <= 0:
        return None
    weights = torch.zeros(num_classes, dtype=torch.float32)
    nonzero = counts > 0
    weights[nonzero] = (total / (float(nonzero.sum().item()) * counts[nonzero])).float()
    return weights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help=(
            "Train on pre-computed frozen-backbone features instead of raw "
            "images.  Expects <dir>/train and <dir>/eval written by "
            "scripts/cache_trio_features.py.  The heavy backbones are NOT "
            "built in this mode (only the trainable mix + head)."
        ),
    )
    parser.add_argument(
        "--closed-test",
        action="store_true",
        help=(
            "Closed-set sanity check: evaluate on the training set itself "
            "(eval_ds = train_ds).  Useful to verify the head can fit data "
            "it has seen -- if the marquee metric is still NaN here the bug "
            "is structural (head/loss), not generalisation."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.epochs from YAML (handy for overfit sanity tests).",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping (forces full --epochs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override train.seed from YAML (for multi-seed significance runs).",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Warm-start: load model weights from this checkpoint (.pt) with "
            "strict=False before training, so shared params start at the given "
            "solution and newly-added modules keep their init.  Used to test a new "
            "fusion module on the fragile dmg_cls task without the from-scratch "
            "init-lottery collapse."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Override output directory for results JSON and checkpoint.  Files "
            "are named <config_stem>.json / <config_stem>.pt inside this dir, "
            "letting multi-seed runs write to isolated folders."
        ),
    )
    args = parser.parse_args(argv)

    raw = _load_yaml(args.config)
    train_cfg = raw.get("train") or {}
    cfg = _build_poe_fuse_config(raw)
    label_kind = _label_kind(cfg.head.kind)
    spec = _resolve_task_spec(train_cfg, cfg.head.kind)

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 42))
    train_cfg["seed"] = seed
    if args.seed is not None:
        for _k in ("wandb_run_name", "model_name"):
            if train_cfg.get(_k):
                train_cfg[_k] = f"{train_cfg[_k]}-seed{seed}"
    _seed_everything(seed)
    device_str = str(train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(
        device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    # Official TEOChatlas protocol:
    #   - train:  ``train/instruct.json`` filtered to the (dataset, task) of the
    #             current spec (e.g. S2Looking + change_detection_detection).
    #   - eval:   the full ``eval/<task>.json`` (no held-out split, matches the
    #             TEOChat paper's reported numbers).
    from_cache = args.from_cache is not None
    max_records = args.max_records if args.max_records is not None else train_cfg.get("max_records")
    collate_fn = cached_collate if from_cache else _collate

    if from_cache:
        cache_root = Path(args.from_cache)
        train_dataset = CachedTrioDataset(cache_root / "train")
        eval_dataset = CachedTrioDataset(cache_root / "eval")
        if train_dataset.label_kind != label_kind:
            raise ValueError(
                f"cache label_kind={train_dataset.label_kind!r} != config "
                f"label_kind={label_kind!r} (wrong --from-cache dir for this config?)"
            )
        print(
            f"[from-cache] train n={len(train_dataset)} eval n={len(eval_dataset)} "
            f"kind={label_kind} root={cache_root}",
            flush=True,
        )
    else:
        train_src_json = Path(str(train_cfg["train_src_json"]))
        eval_src_json = Path(str(train_cfg["src_json"]))
        image_base = train_cfg.get("image_base") or list(DEFAULT_IMAGE_BASE_CANDIDATES)

        train_dataset = ImagePairDataset(
            train_src_json,
            image_size=cfg.image_size,
            image_base=[Path(p) for p in image_base],
            max_records=max_records,
            label_kind=label_kind,
            text_max_len=cfg.head.text_max_len,
            filter_dataset=spec.train_filter_dataset,
            filter_task=spec.task,
            cls_vocab=spec.cls_vocab,
            min_frames=spec.min_frames,
        )
        print(
            f"train n={len(train_dataset)} skipped={len(train_dataset.skipped)} "
            f"kind={label_kind} (filter dataset={spec.train_filter_dataset!r} task={spec.task!r} "
            f"src={train_src_json})",
            flush=True,
        )
        eval_dataset = ImagePairDataset(
            eval_src_json,
            image_size=cfg.image_size,
            image_base=[Path(p) for p in image_base],
            max_records=max_records,
            label_kind=label_kind,
            text_max_len=cfg.head.text_max_len,
            cls_vocab=spec.cls_vocab,
            min_frames=spec.min_frames,
        )
        print(
            f"eval  n={len(eval_dataset)} skipped={len(eval_dataset.skipped)} "
            f"kind={label_kind} (src={eval_src_json})",
            flush=True,
        )

    train_idx = list(range(len(train_dataset)))
    if args.limit_train is not None:
        train_idx = train_idx[: args.limit_train]
    train_ds = Subset(train_dataset, train_idx)
    if args.closed_test:
        eval_ds = train_ds
        print(f"closed-test: eval = train (n={len(train_idx)})", flush=True)
    else:
        eval_ds = eval_dataset
        eval_ds = _filter_sre_eval(eval_ds, train_cfg)

    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 0))
    use_sre_balance = train_cfg.get("task") == "spatial_referring_expression" and bool(
        train_cfg.get("sre_balanced_batch", True)
    )
    if use_sre_balance:
        pos_idx, neg_idx = _sre_referent_split(train_ds)
        batch_sampler = _SREBalancedBatchSampler(pos_idx, neg_idx, batch_size)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
        print(
            f"[sre-train-balance] referent={len(pos_idx)} empty={len(neg_idx)} "
            f"batch={batch_size} ({batch_size // 2}+{batch_size - batch_size // 2})",
            flush=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    model = PoEFuse(cfg, build_backbones=not from_cache).to(device)
    if getattr(args, "init_from", None) is not None:
        st = torch.load(args.init_from, map_location=device, weights_only=False)
        st = st.get("model", st) if isinstance(st, dict) else st
        missing, unexpected = model.load_state_dict(st, strict=False)
        print(
            f"[warm-start] loaded {args.init_from} strict=False "
            f"(missing={len(missing)} kept-at-init, unexpected={len(unexpected)})",
            flush=True,
        )
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"model params total={model.num_parameters()} trainable={model.num_parameters(trainable_only=True)} head={cfg.head.kind} from_cache={from_cache}",
        flush=True,
    )

    if isinstance(model.head, ReferringSegHead):
        pos_idx, neg_idx = _sre_referent_split(train_ds)
        pw = float(cfg.head.seg_presence_pos_weight)
        if pw <= 0:
            pw = 1.0 if use_sre_balance else len(neg_idx) / max(1, len(pos_idx))
        model.head.set_presence_pos_weight(pw)
        print(
            f"[sre-presence] pos_weight={pw:.3f} (empty={len(neg_idx)} referent={len(pos_idx)})",
            flush=True,
        )

    # xBD damage classification is heavily skewed to "no damage"; without
    # re-weighting the classifier collapses to the majority class (the
    # inverse-prevalence-weighted F1 then stays near zero).  Install
    # inverse-frequency CE weights computed from the training distribution
    # (disable with ``train.class_balanced: false``).
    if cfg.head.kind == "classifier" and bool(train_cfg.get("class_balanced", True)):
        weights = _class_weights(train_dataset, num_classes=model.head.num_classes)
        if weights is not None:
            model.head.set_class_weights(weights.to(device))
            print(f"class-balanced CE weights={weights.tolist()}", flush=True)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 1))
    early_stop_patience = 0 if args.no_early_stop else int(train_cfg.get("early_stop_patience", 5))

    # --- Training-regime stabilizer (opt-in via train cfg) -------------------
    # grad accumulation (smoother gradients for the fragile imbalanced bs=2 head),
    # cosine LR + warmup, and weight-EMA (eval/checkpoint on EMA).  Defaults are
    # no-ops so other tasks are unaffected.
    accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    lr_sched = str(train_cfg.get("lr_schedule", "") or "")
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    opt_steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_opt_steps = opt_steps_per_epoch * max(1, epochs)
    scheduler = None
    if lr_sched == "cosine":
        scheds, milestones = [], []
        if warmup_steps > 0:
            scheds.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_steps
                )
            )
            milestones.append(warmup_steps)
        scheds.append(
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_opt_steps - warmup_steps), eta_min=0.0
            )
        )
        scheduler = (
            scheds[0]
            if len(scheds) == 1
            else torch.optim.lr_scheduler.SequentialLR(optimizer, scheds, milestones=milestones)
        )
    ema = (
        {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if ema_decay > 0
        else None
    )

    def _ema_state_dict():
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for n, e in ema.items():
            sd[n] = e.clone()
        return sd

    def _eval_weights():
        """Run eval (on EMA weights if EMA is on, restoring live weights after)."""
        if ema is None:
            return _run_eval(model, eval_loader, device, cfg=cfg, spec=spec, from_cache=from_cache)
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(_ema_state_dict(), strict=True)
        rep = _run_eval(model, eval_loader, device, cfg=cfg, spec=spec, from_cache=from_cache)
        model.load_state_dict(live, strict=True)
        return rep

    wandb_run, wandb_url = _maybe_init_wandb(train_cfg, cfg, config_path=args.config)

    if args.out_dir is not None:
        ckpt_path = Path(args.out_dir) / f"{args.config.stem}.pt"
    else:
        ckpt_path = Path(str(train_cfg.get("checkpoint", "checkpoints/poe_fuse/run.pt")))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_ckpt_tmp = ckpt_path.with_suffix(".best.tmp.pt")

    history: list[dict[str, Any]] = []
    best_eval_loss = float("inf")
    best_metric = float("-inf")
    best_epoch = -1
    no_improve = 0
    stopped_early = False
    # Select / early-stop on the marquee metric (higher is better); eval_loss
    # is decoupled from F1 (a low-loss epoch can have worse F1), so metric-based
    # selection is both fairer across ablations and matches what we report.
    # Pre-loop floor: eval the (warm-started) weights BEFORE any update so the
    # best checkpoint can never be worse than the starting solution.
    if args.init_from is not None or ema is not None:
        pre = _eval_weights()["metrics"]
        pre_score = float(pre.get(spec.metric, float("nan")))
        if pre_score == pre_score:
            best_metric = pre_score
            best_eval_loss = float(pre.get("eval_loss", float("inf")))
            best_epoch = -1
            torch.save(
                {
                    "model": (_ema_state_dict() if ema is not None else model.state_dict()),
                    "epoch": -1,
                    "eval_metrics": pre,
                },
                best_ckpt_tmp,
            )
            print(f"  [pre-loop floor] {spec.metric}={pre_score:.4f}", flush=True)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_n = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            out = _train_forward(
                model,
                batch,
                device=device,
                spec=spec,
                from_cache=from_cache,
            )
            loss = out["loss"]
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    with torch.no_grad():
                        for n_, p_ in model.named_parameters():
                            if p_.requires_grad and n_ in ema:
                                ema[n_].mul_(ema_decay).add_(p_.detach(), alpha=1.0 - ema_decay)
            n = len(batch["sample_ids"])
            running_loss += float(loss.detach()) * n
            running_n += n
            if step % 10 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} loss={float(loss.detach()):.4f}",
                    flush=True,
                )
        train_loss = running_loss / max(1, running_n)
        eval_report = _eval_weights()
        eval_metrics = eval_report["metrics"]
        marquee = float(eval_metrics.get(spec.metric, float("nan")))
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"eval_loss={eval_metrics.get('eval_loss', float('nan')):.4f} "
            f"{spec.metric}={marquee:.4f}",
            flush=True,
        )

        eval_loss = float(eval_metrics.get("eval_loss", float("inf")))
        score = marquee if marquee == marquee else float("-inf")  # NaN-safe
        if score > best_metric + 1e-6:
            best_metric = score
            best_eval_loss = eval_loss
            best_epoch = epoch
            no_improve = 0
            best_model_state = _ema_state_dict() if ema is not None else model.state_dict()
            torch.save(
                {"model": best_model_state, "epoch": epoch, "eval_metrics": eval_metrics},
                best_ckpt_tmp,
            )
            print(
                f"  [best] epoch={epoch} {spec.metric}={score:.4f} eval_loss={eval_loss:.4f}",
                flush=True,
            )
        else:
            no_improve += 1
            print(
                f"  [no improve] {no_improve}/{early_stop_patience} (best {spec.metric}={best_metric:.4f} @epoch={best_epoch})",
                flush=True,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "n_train_examples": running_n,
                "no_improve": no_improve,
                "best_epoch": best_epoch,
                "best_eval_loss": best_eval_loss,
                **eval_metrics,
            }
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "early_stop/no_improve": no_improve,
                    "early_stop/best_eval_loss": best_eval_loss,
                    "early_stop/best_epoch": best_epoch,
                    **{
                        f"eval/{k}": v
                        for k, v in eval_metrics.items()
                        if isinstance(v, (int, float))
                    },
                }
            )

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(
                f"  [early stop] eval_loss did not improve for {no_improve} epochs "
                f"(patience={early_stop_patience}); stopping at epoch={epoch} "
                f"(best epoch={best_epoch}, eval_loss={best_eval_loss:.4f})",
                flush=True,
            )
            stopped_early = True
            break

    if best_ckpt_tmp.exists():
        state = torch.load(best_ckpt_tmp, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(
            f"restored best weights from epoch={state['epoch']} (eval_loss={best_eval_loss:.4f})",
            flush=True,
        )
        best_ckpt_tmp.unlink(missing_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "backbone": asdict(cfg.backbone),
                "mamba3": asdict(cfg.mamba3),
                "head": asdict(cfg.head),
            },
            "labels": list(XBD_DAMAGE_LABELS),
            "head_kind": cfg.head.kind,
            "train_cfg": train_cfg,
            "history": history,
        },
        ckpt_path,
    )

    if args.out_dir is not None:
        out_json = Path(args.out_dir) / f"{args.config.stem}.json"
    else:
        out_json = Path(str(train_cfg.get("out_json", "results/poe_fuse/run.json")))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    final_eval = _run_eval(
        model,
        eval_loader,
        device,
        cfg=cfg,
        spec=spec,
        from_cache=from_cache,
    )
    out_payload = {
        "model_name": train_cfg.get("model_name", "PoEFuse+PairDiff"),
        "dataset_key": train_cfg.get("dataset_key"),
        "task": train_cfg.get("task"),
        "head_kind": cfg.head.kind,
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "history": history,
        "final_eval": final_eval,
        "labels": list(XBD_DAMAGE_LABELS),
        "n_train": len(train_dataset),
        "n_eval": len(eval_dataset),
        "n_skipped_train": len(getattr(train_dataset, "skipped", []) or []),
        "n_skipped_eval": len(getattr(eval_dataset, "skipped", []) or []),
        "wandb_run_url": wandb_url,
        "early_stop": {
            "patience": early_stop_patience,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_eval_loss": best_eval_loss,
            "epochs_run": len(history),
            "epochs_configured": epochs,
        },
    }
    out_json.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = out_json.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "metrics": final_eval["metrics"],
                "head_kind": cfg.head.kind,
                "wandb_run_url": wandb_url,
                "best_epoch": best_epoch,
                "best_eval_loss": best_eval_loss,
                "stopped_early": stopped_early,
                "epochs_run": len(history),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote out={out_json} summary={summary_path} ckpt={ckpt_path}", flush=True)
    if wandb_run is not None:
        wandb_run.log(
            {
                "final/" + k: v
                for k, v in final_eval["metrics"].items()
                if isinstance(v, (int, float))
            }
        )
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
