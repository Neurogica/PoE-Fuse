"""On-disk cache of the frozen trio-backbone penultimate features.

The trio backbones (DINOv3 ViT-7B/16, SAM 3, Gemma-4-E4B-it) are frozen, so
their penultimate features for a given image never change.  Re-running them
every epoch dominates training cost, so we precompute them **once** and cache
them to disk; the cheap trainable part (per-branch ``Linear`` + Mamba-3 +
head) then trains on the cached tensors.  This is the central enabler that
makes the multi-task PoE-Fuse plan fit a single GPU / short timeline.

Layout::

    <cache_dir>/<split>/manifest.json
    <cache_dir>/<split>/000000.pt
    <cache_dir>/<split>/000001.pt
    ...

Each ``<idx>.pt`` holds one sample::

    {
        "sample_id": str,
        "features": {"dino": (T,196,4096), "sam3": (T,200,256), "gemma": (T,256,2560)},  # fp16
        "image_mask": (T,) bool,
        "label_kind": "cls" | "bbox" | "text",
        "ground_truth": str,
        "question": str,
        "polygons": list[str],
        # exactly one of:
        "label_id": int            # cls
        "bboxes": list[[x1,y1,x2,y2]]   # bbox
        "text_ids": list[int]      # text
    }

``CachedTrioDataset`` + :func:`cached_collate` reproduce the batch contract
that :func:`poe_fuse.train._collate` emits, except the heavy ``images`` tensor is
replaced by a per-branch ``features`` dict consumed by
:meth:`PoEFuseCodec.mix` / :meth:`PoEFuse.forward_features`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .codec import BRANCH_KEYS

MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------- writer


@torch.inference_mode()
def write_split_cache(
    *,
    codec,
    loader,
    spec,
    out_dir: Path,
    device: torch.device,
    resolve_sam3_prompts,
    store_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Run the frozen backbones over ``loader`` and dump per-sample features.

    Args:
        codec: a :class:`PoEFuseCodec` built with ``build_backbones=True``.
        loader: a ``DataLoader`` yielding ``poe_fuse.train._collate`` batches.
        spec: the :class:`DatasetTaskSpec` (for SAM 3 prompt + label kind).
        out_dir: ``<cache_dir>/<split>`` directory (created if missing).
        resolve_sam3_prompts: callable ``(spec, questions) -> list[str]``
            (reuse ``poe_fuse.train._resolve_sam3_prompts`` to stay consistent).
        store_dtype: dtype features are cast to before saving (fp16 default).

    Returns the manifest dict (also written to ``out_dir/manifest.json``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    label_ids: list[int] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        image_mask = batch.get("image_mask")
        if image_mask is not None:
            image_mask = image_mask.to(device, non_blocking=True)
        questions = batch["questions"]
        sam3_prompts = resolve_sam3_prompts(spec, questions)
        raw = codec.encode_backbones(images, questions, sam3_prompts, image_mask=image_mask)
        B = images.size(0)
        kind = batch["label_kind"]
        for b in range(B):
            valid = raw["image_mask"][b].to(torch.bool).cpu()
            sample: dict[str, Any] = {
                "sample_id": batch["sample_ids"][b],
                "features": {
                    key: raw[key][b].to(store_dtype).cpu().contiguous() for key in BRANCH_KEYS
                },
                "image_mask": valid,
                "label_kind": kind,
                "ground_truth": batch["ground_truths"][b],
                "question": questions[b],
                "polygons": list(batch["polygons"][b]),
            }
            if kind == "cls":
                lid = int(batch["labels"][b].item())
                sample["label_id"] = lid
                label_ids.append(lid)
            elif kind == "bbox":
                sample["bboxes"] = [list(map(float, box)) for box in batch["bbox_targets"][b]]
            elif kind == "text":
                sample["text_ids"] = [int(t) for t in batch["text_ids"][b].tolist()]
            else:
                raise ValueError(f"unsupported label_kind: {kind}")
            torch.save(sample, out_dir / f"{idx:08d}.pt")
            idx += 1
        print(f"  cached {idx} samples", flush=True)

    manifest = {
        "task_key": spec.key,
        "label_kind": spec.label_kind,
        "n": idx,
        "store_dtype": str(store_dtype).replace("torch.", ""),
        "branch_keys": list(BRANCH_KEYS),
    }
    if label_ids:
        manifest["label_ids"] = label_ids
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


# ---------------------------------------------------------------- dataset


class CachedTrioDataset(Dataset):
    """Read a split directory written by :func:`write_split_cache`.

    ``override_label_kind`` lets a cls-cached dataset serve as text-labelled
    data (e.g. QA tasks trained with a text head instead of a classifier).
    The ground_truth string is converted to text_ids on the fly in collate.
    """

    def __init__(self, split_dir: Path, *, override_label_kind: str | None = None):
        self.split_dir = Path(split_dir)
        manifest_path = self.split_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no cache manifest at {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.label_kind: str = override_label_kind or self.manifest["label_kind"]
        self.files = sorted(p for p in self.split_dir.glob("*.pt"))
        if len(self.files) != int(self.manifest.get("n", len(self.files))):
            raise RuntimeError(
                f"cache size mismatch at {self.split_dir}: manifest n="
                f"{self.manifest.get('n')} but found {len(self.files)} .pt files"
            )
        if not self.files:
            raise RuntimeError(f"empty feature cache at {self.split_dir}")

    @property
    def label_ids(self) -> list[int] | None:
        return self.manifest.get("label_ids")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        if self.label_kind != sample.get("label_kind"):
            sample["label_kind"] = self.label_kind
        return sample


def cached_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate cached samples into a batch consumed by ``PoEFuse.forward_features``.

    Mirrors ``poe_fuse.train._collate`` but carries a per-branch ``features`` dict
    (each ``(B, T_max, N, d)``) instead of raw ``images``; ``T`` is zero-padded
    to the batch max with a matching ``image_mask``.
    """
    kinds = {b["label_kind"] for b in batch}
    if len(kinds) != 1:
        raise RuntimeError(f"mixed label_kinds in batch: {kinds}")
    kind = kinds.pop()
    B = len(batch)
    T_max = max(int(b["image_mask"].numel()) for b in batch)

    features: dict[str, Tensor] = {}
    for key in BRANCH_KEYS:
        ref = batch[0]["features"][key]
        N, d = ref.shape[-2], ref.shape[-1]
        packed = ref.new_zeros(B, T_max, N, d)
        for i, b in enumerate(batch):
            f = b["features"][key]
            packed[i, : f.size(0)] = f
        features[key] = packed

    image_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, b in enumerate(batch):
        m = b["image_mask"].to(torch.bool)
        image_mask[i, : m.numel()] = m

    out: dict[str, Any] = {
        "features": features,
        "image_mask": image_mask,
        "sample_ids": [b["sample_id"] for b in batch],
        "ground_truths": [b["ground_truth"] for b in batch],
        "questions": [str(b.get("question") or "") for b in batch],
        "polygons": [list(b.get("polygons") or []) for b in batch],
        "label_kind": kind,
    }
    if kind == "cls":
        out["labels"] = torch.tensor([int(b["label_id"]) for b in batch], dtype=torch.long)
    elif kind == "bbox":
        out["bbox_targets"] = [list(b["bboxes"]) for b in batch]
    elif kind == "text":
        if "text_ids" in batch[0]:
            out["text_ids"] = torch.tensor([list(b["text_ids"]) for b in batch], dtype=torch.long)
        else:
            from .heads import encode_text

            max_len = 64
            ids_list = [encode_text(str(b["ground_truth"]), max_len=max_len) for b in batch]
            out["text_ids"] = torch.tensor(ids_list, dtype=torch.long)
    else:
        raise ValueError(f"unsupported label_kind: {kind}")
    return out


def features_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    """Build the ``raw`` dict for :meth:`PoEFuseCodec.mix` from a cached batch."""
    raw: dict[str, Tensor] = {
        key: batch["features"][key].to(device, non_blocking=True) for key in BRANCH_KEYS
    }
    raw["image_mask"] = batch["image_mask"].to(device, non_blocking=True)
    return raw


__all__ = [
    "MANIFEST_NAME",
    "CachedTrioDataset",
    "cached_collate",
    "features_to_device",
    "write_split_cache",
]
