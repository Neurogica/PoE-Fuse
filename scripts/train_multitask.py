"""(1) Single shared-trunk, multi-task training over cached features.

One shared codec (projections + spatial align fusion + Mamba-3) with
per-task heads.  Directly answers the generalist-vs-specialist critique:
a *single* lightweight trunk on frozen heterogeneous experts beats an
instruction-tuned VLM across tasks.

Usage:
  python scripts/train_multitask.py \\
    --tasks-file configs/poe_fuse/multitask/tasks.txt \\
    --epochs 30

Or explicit triples ``key:config:cache_dir`` (see ``--tasks``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poe_fuse import (  # noqa: E402
    CachedTrioDataset,
    PoEFuse,
    cached_collate,
)
from poe_fuse.checkpoint_utils import (  # noqa: E402
    load_multitask_checkpoint,
    save_multitask_checkpoint,
)
from poe_fuse.heads import TaskAdapter  # noqa: E402
from poe_fuse.train import (  # noqa: E402
    _build_poe_fuse_config,
    _class_weights,
    _label_kind,
    _resolve_task_spec,
    _run_eval,
    _train_forward,
)


@dataclass
class TaskBundle:
    key: str
    raw: dict[str, Any]
    cfg: Any
    spec: Any
    label_kind: str
    model: PoEFuse
    train_loader: DataLoader
    eval_loader: DataLoader
    config_path: Path
    cache_path: Path
    adapter: TaskAdapter | None = None


def _parse_task_spec(spec_str: str) -> tuple[str, Path, Path]:
    parts = spec_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"task spec must be key:config:cache_dir; got {spec_str!r}")
    return parts[0], Path(parts[1]), Path(parts[2])


def _trunk_signature(cfg: Any) -> tuple:
    """Hashable signature of shared-trunk hyperparams (must match across tasks)."""
    from dataclasses import asdict

    return (
        asdict(cfg.backbone),
        asdict(cfg.mamba3),
        asdict(cfg.fusion),
        asdict(cfg.mixer),
    )


def _build_bundle(key: str, config: Path, cache: Path, device: torch.device) -> TaskBundle:
    raw = yaml.safe_load(config.read_text())
    cfg = _build_poe_fuse_config(raw)
    train_cfg = raw.get("train") or {}
    spec = _resolve_task_spec(train_cfg, cfg.head.kind)
    label_kind = _label_kind(cfg.head.kind)
    model = PoEFuse(cfg, build_backbones=False).to(device)

    override_lk = label_kind if label_kind != _label_kind("classifier") else None
    cache_lk_kwarg = {"override_label_kind": label_kind} if override_lk else {}
    train_ds = CachedTrioDataset(cache / "train", **cache_lk_kwarg)
    eval_ds = CachedTrioDataset(cache / "eval", **cache_lk_kwarg)

    bs = int(train_cfg.get("batch_size", 8))
    nw = int(train_cfg.get("num_workers", 2))
    # persistent_workers keeps the worker pool alive across the per-epoch
    # iter() calls.  Without it, all 6 loaders (3 train + 3 eval) tear down and
    # respawn workers every epoch, which intermittently deadlocks at the epoch
    # boundary (observed: run hangs with GPU idle, new worker PIDs spinning).
    # Reusing workers eliminates the respawn and is training-identical (the
    # sampler order is seeded independently of num_workers).
    persist = nw > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        collate_fn=cached_collate,
        drop_last=False,
        persistent_workers=persist,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=cached_collate,
        persistent_workers=persist,
    )

    if cfg.head.kind == "classifier" and bool(train_cfg.get("class_balanced", True)):
        weights = _class_weights(train_ds, num_classes=model.head.num_classes)
        if weights is not None:
            model.head.set_class_weights(weights.to(device))

    return TaskBundle(
        key=key,
        raw=raw,
        cfg=cfg,
        spec=spec,
        label_kind=label_kind,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config_path=config,
        cache_path=cache,
    )


def _share_trunk(bundles: list[TaskBundle]) -> None:
    shared = bundles[0].model.codec
    for b in bundles[1:]:
        b.model.codec = shared


def _collect_params(bundles: list[TaskBundle]) -> list[torch.nn.Parameter]:
    seen: dict[int, torch.nn.Parameter] = {}
    for b in bundles:
        for p in b.model.parameters():
            if p.requires_grad:
                seen[id(p)] = p
    return list(seen.values())


def _load_tasks(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    if args.tasks_file:
        lines = [
            ln.strip()
            for ln in Path(args.tasks_file).read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        return [_parse_task_spec(ln) for ln in lines]
    if not args.tasks:
        raise SystemExit("provide --tasks or --tasks-file")
    return [_parse_task_spec(t) for t in args.tasks]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", help="key:config_path:cache_dir triples")
    ap.add_argument(
        "--tasks-file",
        type=Path,
        help="line-based task list (see configs/poe_fuse/multitask/tasks.txt)",
    )
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="max grad-norm for gradient clipping (0=off). Stabilizes anchor-free "
        "expert-subset arms: without DINO (the identity-aligned canonical grid) "
        "the bs=2 focal classifier's large gradients (~200) corrupt the shared "
        "trunk and drive the change head to NaN.",
    )
    ap.add_argument(
        "--adapter",
        action="store_true",
        help="Insert a per-task lightweight bottleneck adapter between trunk and head.",
    )
    ap.add_argument(
        "--adapter-bottleneck", type=int, default=0, help="Adapter bottleneck dim (0 = d_model//4)."
    )
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="load --resume checkpoint, evaluate all tasks once, write --out, exit "
        "(no training). Used to re-score existing checkpoints after a metric fix.",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "results/poe_fuse/multitask_trunk.json")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "checkpoints/poe_fuse/multitask_trunk.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    task_specs = _load_tasks(args)
    bundles = [_build_bundle(key, cfg, cache, device) for key, cfg, cache in task_specs]

    sig0 = _trunk_signature(bundles[0].cfg)
    for b in bundles[1:]:
        if _trunk_signature(b.cfg) != sig0:
            raise SystemExit(
                f"trunk mismatch for task {b.key}: backbone/mamba3/fusion/mixer must "
                f"match across all multitask configs (see configs/poe_fuse/multitask/)"
            )

    for b in bundles:
        print(
            f"[task {b.key}] train={len(b.train_loader.dataset)} "
            f"eval={len(b.eval_loader.dataset)} head={b.cfg.head.kind} "
            f"metric={b.spec.metric}",
            flush=True,
        )

    _share_trunk(bundles)
    if args.adapter:
        d = bundles[0].cfg.d_s
        for b in bundles:
            adp = TaskAdapter(d, bottleneck=args.adapter_bottleneck).to(device)
            b.model.adapter = adp
            b.adapter = adp
            print(
                f"[adapter] {b.key}: bottleneck={args.adapter_bottleneck or d // 4} "
                f"params={sum(p.numel() for p in adp.parameters()) / 1e3:.1f}K",
                flush=True,
            )
    if args.resume and args.resume.is_file():
        meta = load_multitask_checkpoint(bundles, args.resume, device=device)
        print(f"[resume] loaded {args.resume} epoch={meta.get('epoch')}", flush=True)

    params = _collect_params(bundles)
    n_trunk = sum(p.numel() for p in bundles[0].model.codec.parameters())
    print(
        f"shared trunk params={n_trunk / 1e6:.1f}M | "
        f"total trainable (dedup)={sum(p.numel() for p in params) / 1e6:.1f}M",
        flush=True,
    )

    if args.eval_only:
        scores: dict[str, float] = {}
        for b in bundles:
            b.model.eval()
            rep = _run_eval(b.model, b.eval_loader, device, cfg=b.cfg, spec=b.spec, from_cache=True)
            scores[b.key] = float(rep["metrics"].get(b.spec.metric, float("nan")))
        mean_score = float(np.nanmean(list(scores.values())))
        print(
            f"[eval-only] mean={mean_score:.4f} | "
            + " ".join(f"{k}={v:.4f}" for k, v in scores.items()),
            flush=True,
        )
        row = {"epoch": -1, "scores": scores, "mean": mean_score, "train_loss": {}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "tasks": [b.key for b in bundles],
                    "best_mean": mean_score,
                    "best_epoch": -1,
                    "best": row,
                    "final": row,
                    "history": [row],
                    "trunk_params_m": n_trunk / 1e6,
                    "checkpoint": str(args.resume),
                    "eval_only": True,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")
        return

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_mean = float("-inf")
    best_epoch = -1
    no_improve = 0
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        for b in bundles:
            b.model.train()
        iters = {b.key: iter(b.train_loader) for b in bundles}
        steps = max(len(b.train_loader) for b in bundles)
        running = {b.key: 0.0 for b in bundles}
        counts = {b.key: 0 for b in bundles}
        for _ in range(steps):
            for b in bundles:
                try:
                    batch = next(iters[b.key])
                except StopIteration:
                    iters[b.key] = iter(b.train_loader)
                    batch = next(iters[b.key])
                out = _train_forward(b.model, batch, device=device, spec=b.spec, from_cache=True)
                loss = out["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                do_step = True
                if args.grad_clip > 0:
                    gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    do_step = bool(torch.isfinite(gnorm))  # skip a NaN/Inf step
                if do_step:
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=True)
                n = len(batch["sample_ids"])
                running[b.key] += float(loss.detach()) * n
                counts[b.key] += n

        scores: dict[str, float] = {}
        for b in bundles:
            b.model.eval()
            rep = _run_eval(b.model, b.eval_loader, device, cfg=b.cfg, spec=b.spec, from_cache=True)
            scores[b.key] = float(rep["metrics"].get(b.spec.metric, float("nan")))
        mean_score = float(np.nanmean(list(scores.values())))
        tl = {k: running[k] / max(1, counts[k]) for k in running}
        row = {"epoch": epoch, "scores": scores, "mean": mean_score, "train_loss": tl}
        history.append(row)
        score_str = " ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"epoch={epoch} mean={mean_score:.4f} | {score_str}", flush=True)

        if mean_score > best_mean + 1e-6:
            best_mean = mean_score
            best_epoch = epoch
            no_improve = 0
            save_multitask_checkpoint(
                bundles,
                args.ckpt,
                epoch=epoch,
                scores=scores,
                extra={"mean": mean_score},
            )
            print(f"  [best] mean={best_mean:.4f} (saved {args.ckpt})", flush=True)
        else:
            no_improve += 1
            if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                print(
                    f"  [early stop] no mean improvement for {no_improve} epochs",
                    flush=True,
                )
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "tasks": [b.key for b in bundles],
        "best_mean": best_mean,
        "best_epoch": best_epoch,
        "best": max(history, key=lambda h: h["mean"]) if history else None,
        "final": history[-1] if history else None,
        "history": history,
        "trunk_params_m": n_trunk / 1e6,
        "checkpoint": str(args.ckpt),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    if summary["best"]:
        print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":
    main()
