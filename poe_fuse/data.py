"""Image-stack dataset for PoEFuse training (all four target tasks).

Reads TEOChatlas-style JSON records (each row has a ``video`` field that,
despite the name, is a list of image paths for the change-detection
tasks) and returns the raw image-stack tensor (variable ``T_i`` per
sample, ``T_i >= 2``) plus a task-shaped label:

* ``label_kind="cls"``   -- ``label_id: int`` (xBD damage classification).
* ``label_kind="bbox"``  -- ``bboxes: list[[x1, y1, x2, y2]]`` in 0-100
                              TEOChatlas coordinates (S2Looking / xBD loc).
* ``label_kind="text"``  -- ``text_ids: list[int]`` already padded to
                              ``HeadConfig.text_max_len`` (Yes/No QA).

The dataset is intentionally cache-free: PIL load + resize is cheap
compared to the dual vision encoder forward pass.  See
.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# fMoW High-Res frames can exceed PIL's default decompression-bomb guard
# (>178M px). These are trusted local dataset files, so lift the limit.
Image.MAX_IMAGE_PIXELS = None

from .heads import PAD_ID, encode_text  # noqa: E402

LabelKind = Literal["cls", "bbox", "text"]


DEFAULT_IMAGE_BASE_CANDIDATES: tuple[str, ...] = (
    "/data/TEOChatlas/eval",
    "/data/TEOChatlas/train",
    "/data/TEOChatlas",
    "/data",
)


XBD_DAMAGE_LABELS: tuple[str, ...] = (
    "no damage",
    "minor damage",
    "major damage",
    "destroyed",
    "unclassified",
)


_BBOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def _normalize_label(text: str) -> str:
    cleaned = re.sub(r"[^a-z ]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def xbd_label_id(answer: str) -> int | None:
    norm = _normalize_label(answer)
    for i, label in enumerate(XBD_DAMAGE_LABELS):
        if norm == label or norm.startswith(label):
            return i
    return None


# fMoW functional-map-of-the-world 62-way scene classification (TEOChatlas
# `scene_classification` task).  Order is fixed so label ids are stable across
# train/eval/cache; the eval scorer maps the predicted id back to this string.
FMOW_SCENE_LABELS: tuple[str, ...] = (
    "airport",
    "airport hangar",
    "airport terminal",
    "amusement park",
    "aquaculture",
    "archaeological site",
    "barn",
    "border checkpoint",
    "burial site",
    "car dealership",
    "construction site",
    "crop field",
    "dam",
    "debris or rubble",
    "educational institution",
    "electric substation",
    "factory or powerplant",
    "fire station",
    "flooded road",
    "fountain",
    "gas station",
    "golf course",
    "ground transportation station",
    "helipad",
    "hospital",
    "impoverished settlement",
    "interchange",
    "lake or pond",
    "lighthouse",
    "military facility",
    "multi-unit residential",
    "nuclear powerplant",
    "office building",
    "oil or gas facility",
    "park",
    "parking lot or garage",
    "place of worship",
    "police station",
    "port",
    "prison",
    "race track",
    "railway bridge",
    "recreational facility",
    "road bridge",
    "runway",
    "shipyard",
    "shopping mall",
    "single-unit residential",
    "smokestack",
    "solar farm",
    "space facility",
    "stadium",
    "storage tank",
    "surface mine",
    "swimming pool",
    "toll booth",
    "tower",
    "tunnel opening",
    "waste disposal",
    "water treatment facility",
    "wind farm",
    "zoo",
)

_FMOW_LABEL_TO_ID = {lab: i for i, lab in enumerate(FMOW_SCENE_LABELS)}


def fmow_label_id(answer: str) -> int | None:
    """Map a free-text fMoW class answer to its stable class id (or None)."""
    norm = _normalize_label(answer)
    if norm in _FMOW_LABEL_TO_ID:
        return _FMOW_LABEL_TO_ID[norm]
    # Tolerate trailing words / punctuation ("airport." etc).
    for lab, i in _FMOW_LABEL_TO_ID.items():
        if norm == lab or norm.startswith(lab + " ") or norm.startswith(lab):
            return i
    return None


# Registry of cls label vocabularies, keyed by the value placed in a config's
# ``head.cls_vocab`` (defaults to xBD damage).  Lets one classifier head serve
# multiple datasets without hard-coding xBD everywhere.
ABCD_LABELS: tuple[str, ...] = ("survived", "washed away")

CDVQA_LABELS: tuple[str, ...] = (
    "0",
    "0_to_10",
    "10_to_20",
    "20_to_30",
    "30_to_40",
    "40_to_50",
    "50_to_60",
    "60_to_70",
    "70_to_80",
    "80_to_90",
    "90_to_100",
    "buildings",
    "low_vegetation",
    "no",
    "nonvegetated ground surface",
    "playgrounds",
    "trees",
    "water",
    "yes",
)

XBD_QA_LABELS: tuple[str, ...] = (
    "a flood.",
    "a hurricane.",
    "a tsunami.",
    "a volcanic eruption.",
    "a wildfire.",
    "an earthquake.",
    "bottom left.",
    "bottom right.",
    "center.",
    "no.",
    "no parts affected.",
    "top.",
    "top left.",
    "top right.",
    "yes.",
)

YES_NO_LABELS: tuple[str, ...] = ("no", "yes")
YES_NO_DOT_LABELS: tuple[str, ...] = ("no.", "yes.")

CLS_VOCABS: dict[str, tuple[str, ...]] = {
    "xbd_damage": XBD_DAMAGE_LABELS,
    "fmow_scene": FMOW_SCENE_LABELS,
    "abcd": ABCD_LABELS,
    "cdvqa": CDVQA_LABELS,
    "xbd_qa": XBD_QA_LABELS,
    "yes_no": YES_NO_LABELS,
    "yes_no_dot": YES_NO_DOT_LABELS,
}

_GENERIC_VOCABS: dict[str, dict[str, int]] = {}


def _get_generic_vocab(vocab: str) -> dict[str, int]:
    if vocab not in _GENERIC_VOCABS:
        labels = CLS_VOCABS.get(vocab)
        if labels is None:
            raise ValueError(f"unknown cls_vocab: {vocab!r}")
        _GENERIC_VOCABS[vocab] = {_normalize_label(lab): i for i, lab in enumerate(labels)}
    return _GENERIC_VOCABS[vocab]


def cls_label_id(answer: str, vocab: str = "xbd_damage") -> int | None:
    if vocab == "fmow_scene":
        return fmow_label_id(answer)
    if vocab == "xbd_damage":
        return xbd_label_id(answer)
    mapping = _get_generic_vocab(vocab)
    norm = _normalize_label(answer)
    if norm in mapping:
        return mapping[norm]
    for k, v in mapping.items():
        if norm.startswith(k) or k.startswith(norm):
            return v
    return None


def parse_bbox_string(answer: str) -> list[list[float]]:
    """Return the list of ``[x1, y1, x2, y2]`` floats in ``answer``.

    Accepts the TEOChatlas convention (0-100 coordinates).  Empty answers
    or ``"no change"`` style strings yield ``[]``.
    """
    return [[float(g) for g in m.groups()] for m in _BBOX_RE.finditer(answer)]


def format_bbox_list(bboxes: Iterable[Iterable[float]]) -> str:
    rendered = ", ".join("[" + ", ".join(f"{c:g}" for c in bbox) + "]" for bbox in bboxes)
    return f"{rendered}." if rendered else ""


def sanitize_bbox_response(text: str) -> str:
    """Drop ``[...]`` groups that don't contain exactly 4 comma-separated floats.

    The vendored TEOChat evaluator (``teochat_eval.detection.evaluate_masks``)
    assumes every bracketed group inside a response is a 4-tuple bbox and
    crashes with ``IndexError`` on shorter / non-numeric brackets that the
    a character-level text decoder can emit early in training.
    """

    def _keep(m: re.Match[str]) -> str:
        try:
            vals = [float(v) for v in m.group(1).split(",")]
        except ValueError:
            return ""
        return m.group(0) if len(vals) == 4 else ""

    return re.sub(r"\[(.*?)\]", _keep, text)


def _polygons_from_row(row: dict[str, Any]) -> list[str]:
    """Return the list of non-empty WKT polygon strings on ``row``.

    TEOChatlas eval JSONs use a top-level ``polygon`` field holding one WKT
    string per ground-truth instance (building / changed building).  QA-only
    records (e.g., Yes/No SRE QA) store a placeholder ``['']``; we drop those.
    """
    raw = row.get("polygon") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(p) for p in raw if isinstance(p, str) and p.strip()]


def _conversation_value(row: dict[str, Any], idx: int) -> str:
    conversations = row.get("conversations") or []
    if (
        isinstance(conversations, list)
        and len(conversations) > idx
        and isinstance(conversations[idx], dict)
    ):
        return str(conversations[idx].get("value") or "").strip()
    return ""


def _question(row: dict[str, Any]) -> str:
    return str(row.get("question") or "").strip() or _conversation_value(row, 0)


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("answer") or "").strip() or _conversation_value(row, 1)


def _sample_id(row: dict[str, Any], idx: int) -> str:
    raw = str(row.get("id") or row.get("sample_id") or row.get("question_id") or idx)
    return raw or str(idx)


def _resolve_image_path(raw: str, image_base: Iterable[Path]) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    candidates: list[Path] = []
    rel = candidate
    while True:
        for base in image_base:
            candidates.append(base / rel)
        if len(rel.parts) <= 1:
            break
        rel = Path(*rel.parts[1:])
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        f"could not resolve image path: {raw}; tried {len(candidates)} candidates"
    )


def _image_paths_from_row(row: dict[str, Any], image_base: list[Path]) -> list[Path]:
    raw_paths = row.get("video") or row.get("images") or row.get("image_paths") or []
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, list):
        raise ValueError(f"unsupported image-paths schema: {type(raw_paths).__name__}")
    return [_resolve_image_path(str(p), image_base) for p in raw_paths]


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0


def records_from_json(src: Path, *, max_records: int | None) -> list[dict[str, Any]]:
    raw = json.loads(Path(src).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected JSON list at {src}; got {type(raw).__name__}")
    rows = [r for r in raw if isinstance(r, dict)]
    return rows if max_records is None else rows[:max_records]


class ImagePairDataset(Dataset):
    """Yield image-pair samples shaped for ``label_kind``."""

    def __init__(
        self,
        src_json: Path,
        *,
        image_size: int = 224,
        image_base: list[Path] | None = None,
        max_records: int | None = None,
        label_kind: LabelKind = "cls",
        text_max_len: int = 96,
        filter_dataset: str | None = None,
        filter_task: str | None = None,
        cls_vocab: str = "xbd_damage",
        min_frames: int = 2,
    ):
        self.src_json = Path(src_json)
        self.image_size = image_size
        self.image_base = [Path(b) for b in (image_base or DEFAULT_IMAGE_BASE_CANDIDATES)]
        self.label_kind = label_kind
        self.text_max_len = text_max_len
        self.cls_vocab = cls_vocab
        # Change-detection tasks need >=2 frames; fMoW scene classification has
        # single-frame records, so the minimum is configurable.
        self.min_frames = int(min_frames)
        # When filters are set we need to look at *all* rows first and slice
        # ``max_records`` after filtering -- otherwise a tiny per-task subset
        # of a 500k-row mixed instruct.json would be silently empty.
        rows = records_from_json(
            self.src_json,
            max_records=None if (filter_dataset or filter_task) else max_records,
        )
        if filter_dataset is not None or filter_task is not None:
            rows = [
                r
                for r in rows
                if (filter_dataset is None or r.get("dataset") == filter_dataset)
                and (filter_task is None or r.get("task") == filter_task)
            ]
            if max_records is not None:
                rows = rows[:max_records]

        self.rows: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            sid = _sample_id(row, idx)
            try:
                paths = _image_paths_from_row(row, self.image_base)
            except FileNotFoundError as exc:
                self.skipped.append({"sample_id": sid, "reason": str(exc)})
                continue
            if len(paths) < self.min_frames:
                self.skipped.append(
                    {
                        "sample_id": sid,
                        "reason": f"need >={self.min_frames} image paths; got {len(paths)}",
                    }
                )
                continue

            answer = _answer(row)
            entry: dict[str, Any] = {
                "sample_id": sid,
                "paths": list(paths),
                "answer": answer,
                "question": _question(row),
                "polygons": _polygons_from_row(row),
                "raw": row,
            }

            if label_kind == "cls":
                label_id = cls_label_id(answer, self.cls_vocab)
                if label_id is None:
                    self.skipped.append(
                        {"sample_id": sid, "reason": f"unparseable cls label: {answer!r}"}
                    )
                    continue
                entry["label_id"] = int(label_id)
            elif label_kind == "bbox":
                entry["bboxes"] = parse_bbox_string(answer)
            elif label_kind == "text":
                ids = encode_text(answer, max_len=text_max_len)
                ids = ids + [PAD_ID] * (text_max_len - len(ids))
                entry["text_ids"] = ids
            else:
                raise ValueError(f"unsupported label_kind: {label_kind!r}")

            self.rows.append(entry)

        if not self.rows:
            raise RuntimeError(
                f"no usable rows in {self.src_json} (kind={label_kind}, skipped={len(self.skipped)})"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        images = torch.stack(
            [_load_image_tensor(p, self.image_size) for p in row["paths"]],
            dim=0,
        )  # (T_i, 3, H, W) with T_i variable per sample.
        item: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "images": images,
            "ground_truth": row["answer"],
            "question": row["question"],
            "polygons": list(row["polygons"]),
            "meta": row["raw"],
            "label_kind": self.label_kind,
        }
        if self.label_kind == "cls":
            item["label_id"] = int(row["label_id"])
        elif self.label_kind == "bbox":
            item["bboxes"] = [list(b) for b in row["bboxes"]]
        elif self.label_kind == "text":
            item["text_ids"] = list(row["text_ids"])
        return item


__all__ = [
    "CLS_VOCABS",
    "DEFAULT_IMAGE_BASE_CANDIDATES",
    "FMOW_SCENE_LABELS",
    "ImagePairDataset",
    "LabelKind",
    "XBD_DAMAGE_LABELS",
    "_polygons_from_row",
    "cls_label_id",
    "fmow_label_id",
    "format_bbox_list",
    "parse_bbox_string",
    "records_from_json",
    "xbd_label_id",
]
