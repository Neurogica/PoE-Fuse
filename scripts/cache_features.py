"""Precompute and cache the frozen trio-backbone features for one task.

The trio backbones (DINOv3 / SAM 3 / Gemma-4) are frozen, so their
penultimate features can be computed once and reused every epoch.  This
script materialises that cache; ``scripts/train_multitask.py`` then trains
the cheap, trainable part (projections + Mamba-3 + head) on it.

Usage::

    uv run scripts/cache_features.py \
        --config configs/xbd_dmg_cls_head_poe_focal.yaml \
        --out-dir feature_cache/xbd_dmg_cls \
        --split both --batch-size 8

Produces ``<out-dir>/train/`` and/or ``<out-dir>/eval/`` directories that
match ``CachedTrioDataset``'s expected layout.  Use ``--limit-train`` /
``--max-records`` for a quick smoke cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from poe_fuse import (  # noqa: E402
    DEFAULT_IMAGE_BASE_CANDIDATES,
    ImagePairDataset,
    PoEFuse,
    write_split_cache,
)
from poe_fuse.train import (  # noqa: E402
    _build_poe_fuse_config,
    _collate,
    _label_kind,
    _load_yaml,
    _resolve_sam3_prompts,
    _resolve_task_spec,
)


def _build_dataset(train_cfg, cfg, *, which: str, spec, max_records):
    label_kind = _label_kind(cfg.head.kind)
    image_base = train_cfg.get("image_base") or list(DEFAULT_IMAGE_BASE_CANDIDATES)
    common = dict(
        image_size=cfg.image_size,
        image_base=[Path(p) for p in image_base],
        max_records=max_records,
        label_kind=label_kind,
        text_max_len=cfg.head.text_max_len,
        cls_vocab=getattr(spec, "cls_vocab", "xbd_damage"),
        min_frames=getattr(spec, "min_frames", 2),
    )
    if which == "train":
        return ImagePairDataset(
            Path(str(train_cfg["train_src_json"])),
            filter_dataset=spec.train_filter_dataset,
            filter_task=spec.task,
            **common,
        )
    return ImagePairDataset(Path(str(train_cfg["src_json"])), **common)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--sample-records",
        type=int,
        default=None,
        help=(
            "Randomly subsample this many records (seeded) instead of taking the "
            "first N.  Use for class-balanced coverage when the source json is "
            "ordered (e.g. fMoW train), where --max-records would miss classes."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--store-dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    args = parser.parse_args(argv)

    raw = _load_yaml(args.config)
    train_cfg = raw.get("train") or {}
    cfg = _build_poe_fuse_config(raw)
    spec = _resolve_task_spec(train_cfg, cfg.head.kind)
    store_dtype = getattr(torch, args.store_dtype)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoEFuse(cfg, build_backbones=True).to(device)
    model.eval()
    print(
        f"backbones built; total params={model.num_parameters()} "
        f"(frozen feature extractor) device={device}",
        flush=True,
    )

    splits = ["train", "eval"] if args.split == "both" else [args.split]
    for which in splits:
        dataset = _build_dataset(
            train_cfg, cfg, which=which, spec=spec, max_records=args.max_records
        )
        if which == "train" and args.sample_records is not None:
            import random as _random

            n = len(dataset)
            k = min(args.sample_records, n)
            rng = _random.Random(args.sample_seed)
            idx = sorted(rng.sample(range(n), k))
            dataset = Subset(dataset, idx)
            print(
                f"[{which}] random subsample {k}/{n} (seed={args.sample_seed})",
                flush=True,
            )
        elif which == "train" and args.limit_train is not None:
            dataset = Subset(dataset, list(range(min(args.limit_train, len(dataset)))))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate,
            num_workers=args.num_workers,
        )
        out_dir = args.out_dir / which
        print(f"[{which}] n={len(dataset)} -> {out_dir}", flush=True)
        manifest = write_split_cache(
            codec=model.codec,
            loader=loader,
            spec=spec,
            out_dir=out_dir,
            device=device,
            resolve_sam3_prompts=_resolve_sam3_prompts,
            store_dtype=store_dtype,
        )
        print(f"[{which}] done: {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
