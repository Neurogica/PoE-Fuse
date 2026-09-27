"""Evaluate a multitask trunk checkpoint on all tasks in tasks.txt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_multitask as mt  # noqa: E402

from poe_fuse.checkpoint_utils import load_multitask_checkpoint  # noqa: E402
from poe_fuse.train import _run_eval  # noqa: E402


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return float(obj)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, torch.Tensor) and v.numel() == 1:
            out[k] = float(v.detach().cpu().item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "results/poe_fuse/multitask_trunk_eval.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ns = argparse.Namespace(tasks_file=args.tasks_file, tasks=None)
    task_specs = mt._load_tasks(ns)
    bundles = [mt._build_bundle(k, c, cache, device) for k, c, cache in task_specs]
    mt._share_trunk(bundles)
    meta = load_multitask_checkpoint(bundles, args.checkpoint, device=device)

    scores = {}
    for b in bundles:
        b.model.eval()
        rep = _run_eval(b.model, b.eval_loader, device, cfg=b.cfg, spec=b.spec, from_cache=True)
        scores[b.key] = {
            "metric": b.spec.metric,
            "value": float(rep["metrics"].get(b.spec.metric, float("nan"))),
            "metrics": _scalar_metrics(rep["metrics"]),
        }
        print(f"{b.key}: {scores[b.key]['value']:.4f}", flush=True)

    meta_slim = {
        "epoch": meta.get("epoch"),
        "scores": _jsonify(meta.get("scores", {})),
        "mean": meta.get("mean"),
    }
    out = {
        "checkpoint": str(args.checkpoint),
        "meta": meta_slim,
        "scores": scores,
        "mean": float(np.nanmean([s["value"] for s in scores.values()])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
