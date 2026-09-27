"""Download the frozen trio-backbone weights into ``models/``.

The PoEFuse codec expects three local checkpoints (see
``src/model/poe_fuse/config.py`` :class:`TrioBackboneConfig`):

* DINOv3 ViT-7B/16 SAT-493M weights -> ``models/dinov3_vit7b16_pretrain_sat493m-a6675841.pth``
* SAM 3 image model                 -> ``models/facebook/sam3/sam3.pt``
* Gemma-4-E4B-it                     -> ``models/google/gemma-4-E4B-it/`` (dir)

ALL THREE ARE GATED on the Hugging Face Hub (Meta / Google license
approval).  You must:

1. Request + be granted access on each model page while logged in to the
   same HF account.
2. Provide a token with `export HF_TOKEN=hf_xxx` (or `hf auth login`).

Then run::

    uv run scripts/download_backbones.py --which sam3 dinov3 gemma

``gemma`` is usually already in the HF cache; the script will materialise a
``models/google/gemma-4-E4B-it`` symlink/dir from it.  Use ``--which`` to pick
a subset.  Nothing here re-downloads files already present in the HF cache.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = REPO_ROOT / "models"

# HF repo coordinates.  Adjust the DINOv3 filename/repo if Meta re-publishes
# under a different id; the codec only cares about the final local path.
SAM3_REPO = "facebook/sam3"
SAM3_FILES = ["sam3.pt"]
DINOV3_REPO = "facebook/dinov3-vit7b16-pretrain-sat493m"
DINOV3_WEIGHT_FILE = "dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
GEMMA_REPO = "google/gemma-4-E4B-it"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def download_sam3() -> None:
    from huggingface_hub import hf_hub_download

    dest = MODELS / "facebook" / "sam3"
    dest.mkdir(parents=True, exist_ok=True)
    for fname in SAM3_FILES:
        path = hf_hub_download(repo_id=SAM3_REPO, filename=fname, token=_token())
        link = dest / fname
        link.unlink(missing_ok=True)
        link.symlink_to(path)
        print(f"[sam3] {fname} -> {link} ({path})", flush=True)


def download_dinov3() -> None:
    from huggingface_hub import hf_hub_download

    MODELS.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=DINOV3_REPO, filename=DINOV3_WEIGHT_FILE, token=_token())
    link = MODELS / DINOV3_WEIGHT_FILE
    link.unlink(missing_ok=True)
    link.symlink_to(path)
    print(f"[dinov3] {DINOV3_WEIGHT_FILE} -> {link} ({path})", flush=True)


def download_gemma() -> None:
    from huggingface_hub import snapshot_download

    snap = snapshot_download(repo_id=GEMMA_REPO, token=_token())
    dest = MODELS / "google" / "gemma-4-E4B-it"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True) if dest.is_symlink() else None
    if not dest.exists():
        dest.symlink_to(snap)
    print(f"[gemma] -> {dest} ({snap})", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--which",
        nargs="+",
        choices=["sam3", "dinov3", "gemma"],
        default=["sam3", "dinov3", "gemma"],
    )
    args = parser.parse_args(argv)

    if _token() is None:
        print(
            "WARNING: no HF_TOKEN / HUGGING_FACE_HUB_TOKEN set. The Meta/Google "
            "models are gated and the download will 401 unless your account has "
            "been granted access and a token is provided.",
            flush=True,
        )

    fns = {"sam3": download_sam3, "dinov3": download_dinov3, "gemma": download_gemma}
    failures: list[str] = []
    for name in args.which:
        try:
            fns[name]()
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append(name)
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
    if failures:
        print(f"\nincomplete: {', '.join(failures)} (see errors above)", flush=True)
        return 1
    print("\nall requested backbones present.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
