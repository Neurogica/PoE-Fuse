"""Task-specialist change-detection / segmentation baselines (pure PyTorch).

Why pure PyTorch + train-from-scratch: the TEOChat-cited specialists are
FC-Siam-Diff (S2Looking change det.) and the xView2 U-Net baseline (xBD
localization). Their official weights ship as old Keras / mmcv stacks that do
not run on recent GPUs. We instead reimplement the standard architectures and
train on the TEOChatlas train split, then evaluate on the eval split, scoring
every baseline identically (per-pixel F1 over 256x256 masks).

Ground truth for training is rasterised from the answer bounding boxes (the
train split has empty polygon fields), in the same [0,100]-normalised frame
the scorer uses. Multi-seed training gives the mean +/- std the paper lacks.

    python scripts/cd_baselines.py \
        --task s2_det --arch fc_siam_diff --seed 0 --epochs 15 --batch-size 16 \
        --out-dir results/specialists/s2_det
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None
DATA_BASE = os.environ.get("TEOCHATLAS_BASE", "/data")
HW = 256
BOX_RE = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")

TASKS = {
    "s2_det": ("S2Looking", "change_detection_detection", "S2Looking_Change_Detection"),
    "xbd_loc": ("xBD", "change_detection_localization", "xBD_Change_Detection_Localization"),
}
OUT_NAME = "TEOChat_prompt_strategy_interleave_chronological_prefix_True.json"


def parse_boxes(text: str) -> list[list[int]]:
    return [list(map(int, m.groups())) for m in BOX_RE.finditer(text)]


def rasterize(boxes, hw: int = HW) -> np.ndarray:
    m = np.zeros((hw, hw), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        xa, xb = sorted((int(x1 / 100 * hw), int(x2 / 100 * hw)))
        ya, yb = sorted((int(y1 / 100 * hw), int(y2 / 100 * hw)))
        m[ya:yb, xa:xb] = 1
    return m


def load_img(path: str) -> np.ndarray:
    with Image.open(os.path.join(DATA_BASE, path)) as im:
        im = im.convert("RGB").resize((HW, HW), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


class CDDataset(Dataset):
    def __init__(self, rows, train: bool):
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        a = load_img(r["video"][0])
        b = load_img(r["video"][1])
        pair = np.concatenate([a, b], axis=2).transpose(2, 0, 1)  # [6,H,W]
        gt_text = r["conversations"][1]["value"]
        mask = rasterize(parse_boxes(gt_text)) if "[" in gt_text else np.zeros((HW, HW), np.uint8)
        return torch.from_numpy(pair), torch.from_numpy(mask).float(), i


# ----------------------------------------------------------------- models
def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
    )


class FCSiamDiff(nn.Module):
    """FC-Siam-Diff (Daudt et al. 2018): shared encoder, decode feature diffs."""

    def __init__(self, ch=3, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3, self.e4 = (
            conv_block(ch, c[0]),
            conv_block(c[0], c[1]),
            conv_block(c[1], c[2]),
            conv_block(c[2], c[3]),
        )
        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = conv_block(c[3] + c[2], c[2])
        self.d2 = conv_block(c[2] + c[1], c[1])
        self.d1 = conv_block(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def encode(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        return s1, s2, s3, s4

    def forward(self, x):
        a, b = x[:, :3], x[:, 3:]
        a1, a2, a3, a4 = self.encode(a)
        b1, b2, b3, b4 = self.encode(b)
        d1, d2, d3, d4 = (
            torch.abs(a1 - b1),
            torch.abs(a2 - b2),
            torch.abs(a3 - b3),
            torch.abs(a4 - b4),
        )
        x = self.d3(torch.cat([self.up(d4), d3], 1))
        x = self.d2(torch.cat([self.up(x), d2], 1))
        x = self.d1(torch.cat([self.up(x), d1], 1))
        return self.out(x).squeeze(1)


class UNet(nn.Module):
    """Plain U-Net on the bitemporal stack (6ch) for building localization."""

    def __init__(self, ch=6, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3, self.e4 = (
            conv_block(ch, c[0]),
            conv_block(c[0], c[1]),
            conv_block(c[1], c[2]),
            conv_block(c[2], c[3]),
        )
        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = conv_block(c[3] + c[2], c[2])
        self.d2 = conv_block(c[2] + c[1], c[1])
        self.d1 = conv_block(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        x = self.d3(torch.cat([self.up(s4), s3], 1))
        x = self.d2(torch.cat([self.up(x), s2], 1))
        x = self.d1(torch.cat([self.up(x), s1], 1))
        return self.out(x).squeeze(1)


class _VGGBlock(nn.Module):
    def __init__(self, ci, cm, co):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(ci, cm, 3, padding=1),
            nn.BatchNorm2d(cm),
            nn.ReLU(inplace=True),
            nn.Conv2d(cm, co, 3, padding=1),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.c(x)


class SNUNet(nn.Module):
    """SNUNet-CD (Fang et al., IEEE GRSL 2021): Siamese NestedUNet (UNet++) with
    dense skip connections + a channel-attention (ECAM) fusion of the multi-level
    change features.  Trained from scratch, same protocol as the other baselines."""

    def __init__(self, ch=3, base=16):
        super().__init__()
        n = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.e0, self.e1 = _VGGBlock(ch, n[0], n[0]), _VGGBlock(n[0], n[1], n[1])
        self.e2, self.e3 = _VGGBlock(n[1], n[2], n[2]), _VGGBlock(n[2], n[3], n[3])
        self.e4 = _VGGBlock(n[3], n[4], n[4])
        # nested decoder nodes; siamese encoder nodes carry 2*n[i] (concat of A,B)
        self.c01 = _VGGBlock(2 * n[0] + 2 * n[1], n[0], n[0])
        self.c11 = _VGGBlock(2 * n[1] + 2 * n[2], n[1], n[1])
        self.c21 = _VGGBlock(2 * n[2] + 2 * n[3], n[2], n[2])
        self.c31 = _VGGBlock(2 * n[3] + 2 * n[4], n[3], n[3])
        self.c02 = _VGGBlock(2 * n[0] + n[0] + n[1], n[0], n[0])
        self.c12 = _VGGBlock(2 * n[1] + n[1] + n[2], n[1], n[1])
        self.c22 = _VGGBlock(2 * n[2] + n[2] + n[3], n[2], n[2])
        self.c03 = _VGGBlock(2 * n[0] + 2 * n[0] + n[1], n[0], n[0])
        self.c13 = _VGGBlock(2 * n[1] + 2 * n[1] + n[2], n[1], n[1])
        self.c04 = _VGGBlock(2 * n[0] + 3 * n[0] + n[1], n[0], n[0])
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(4 * n[0], n[0], 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n[0], 4 * n[0], 1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(4 * n[0], 1, 1)

    def _enc(self, x):
        x0 = self.e0(x)
        x1 = self.e1(self.pool(x0))
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        return x0, x1, x2, x3, x4

    def forward(self, x):
        a, b = x[:, :3], x[:, 3:]
        a0, a1, a2, a3, a4 = self._enc(a)
        b0, b1, b2, b3, b4 = self._enc(b)
        x00, x10, x20 = torch.cat([a0, b0], 1), torch.cat([a1, b1], 1), torch.cat([a2, b2], 1)
        x30, x40 = torch.cat([a3, b3], 1), torch.cat([a4, b4], 1)
        x01 = self.c01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.c11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.c21(torch.cat([x20, self.up(x30)], 1))
        x31 = self.c31(torch.cat([x30, self.up(x40)], 1))
        x02 = self.c02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.c12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.c22(torch.cat([x20, x21, self.up(x31)], 1))
        x03 = self.c03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        x13 = self.c13(torch.cat([x10, x11, x12, self.up(x22)], 1))
        x04 = self.c04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))
        out = torch.cat([x01, x02, x03, x04], 1)
        out = out * self.ca(out)
        return self.out(out).squeeze(1)


class _MixAtt(nn.Module):
    """TinyCD-style mixing+attention block: fuse [a, b, |a-b|] then spatial gate."""

    def __init__(self, c):
        super().__init__()
        self.mix = nn.Sequential(nn.Conv2d(3 * c, c, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, a, b):
        x = self.mix(torch.cat([a, b, torch.abs(a - b)], 1))
        return x * self.gate(x)


class TinyCD(nn.Module):
    """TinyCD-style (Codegoni et al. 2023) lightweight change detector: shared
    light encoder, per-level mix-and-attention of the bitemporal features, tiny
    decoder.  Reimplemented and trained from scratch on the same split."""

    def __init__(self, ch=3, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3, self.e4 = (
            conv_block(ch, c[0]),
            conv_block(c[0], c[1]),
            conv_block(c[1], c[2]),
            conv_block(c[2], c[3]),
        )
        self.pool = nn.MaxPool2d(2)
        self.up = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        self.m1, self.m2, self.m3, self.m4 = (
            _MixAtt(c[0]),
            _MixAtt(c[1]),
            _MixAtt(c[2]),
            _MixAtt(c[3]),
        )
        self.d3 = conv_block(c[3] + c[2], c[2])
        self.d2 = conv_block(c[2] + c[1], c[1])
        self.d1 = conv_block(c[1] + c[0], c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def _enc(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        return s1, s2, s3, s4

    def forward(self, x):
        a, b = x[:, :3], x[:, 3:]
        a1, a2, a3, a4 = self._enc(a)
        b1, b2, b3, b4 = self._enc(b)
        m1, m2, m3, m4 = self.m1(a1, b1), self.m2(a2, b2), self.m3(a3, b3), self.m4(a4, b4)
        x = self.d3(torch.cat([self.up(m4), m3], 1))
        x = self.d2(torch.cat([self.up(x), m2], 1))
        x = self.d1(torch.cat([self.up(x), m1], 1))
        return self.out(x).squeeze(1)


class _XFBlock(nn.Module):
    """Transformer block with spatial-reduction attention (SegFormer-style)."""

    def __init__(self, dim, heads=4, sr=1):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.sr = nn.Conv2d(dim, dim, sr, stride=sr) if sr > 1 else None
        self.srn = nn.LayerNorm(dim) if sr > 1 else None
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.n1(x)
        if self.sr is not None:
            k = q.transpose(1, 2).reshape(B, C, H, W)
            k = self.sr(k).reshape(B, C, -1).transpose(1, 2)
            k = self.srn(k)
        else:
            k = q
        x = x + self.attn(q, k, k, need_weights=False)[0]
        x = x + self.mlp(self.n2(x))
        return x


class ChangeFormer(nn.Module):
    """ChangeFormer-style (Bandara & Patel 2022): hierarchical siamese transformer
    encoder, per-stage bitemporal difference, lightweight all-MLP decoder.
    Reimplemented and trained from scratch on the same split."""

    def __init__(self, ch=3, dims=(32, 64, 128, 256), srs=(8, 4, 2, 1), dec=128):
        super().__init__()
        self.embed = nn.ModuleList(
            [
                nn.Conv2d(
                    ch if i == 0 else dims[i - 1],
                    dims[i],
                    7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    padding=3 if i == 0 else 1,
                )
                for i in range(4)
            ]
        )
        self.norm = nn.ModuleList([nn.LayerNorm(d) for d in dims])
        self.blocks = nn.ModuleList([_XFBlock(dims[i], 4, srs[i]) for i in range(4)])
        self.proj = nn.ModuleList([nn.Conv2d(dims[i], dec, 1) for i in range(4)])
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * dec, dec, 1), nn.BatchNorm2d(dec), nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(dec, 1, 1)

    def _enc(self, x):
        feats = []
        for i in range(4):
            x = self.embed[i](x)
            B, C, H, W = x.shape
            t = x.flatten(2).transpose(1, 2)
            t = self.blocks[i](t, H, W)
            t = self.norm[i](t)
            x = t.transpose(1, 2).reshape(B, C, H, W)
            feats.append(x)
        return feats

    def forward(self, x):
        a, b = x[:, :3], x[:, 3:]
        fa, fb = self._enc(a), self._enc(b)
        H0, W0 = fa[0].shape[2:]
        diffs = []
        for i in range(4):
            d = torch.abs(fa[i] - fb[i])
            d = self.proj[i](d)
            d = F.interpolate(d, size=(H0, W0), mode="bilinear", align_corners=False)
            diffs.append(d)
        x = self.fuse(torch.cat(diffs, 1))
        x = self.out(x)
        return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False).squeeze(1)


# ----------------------------------------------------------------- ChangeMamba
try:
    from mamba_ssm import Mamba as _Mamba
except Exception:  # pragma: no cover - only needed for --arch changemamba
    _Mamba = None


class _VSSBlock(nn.Module):
    """Vision-Mamba block: a bidirectional selective scan (forward + reversed) over
    the row-major flattened feature map, then an MLP.  The bidirectional scan is
    the visual-Mamba (Vim / VMamba family) adaptation of Mamba to 2-D features;
    the selective-scan weights are shared across the two directions."""

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = _Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def _bidir(self, t):  # t: [B, L, C]
        return 0.5 * (self.mamba(t) + torch.flip(self.mamba(torch.flip(t, [1])), [1]))

    def forward(self, x):  # [B, C, H, W]
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self._bidir(self.norm(t))
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


class _STMamba(nn.Module):
    """Spatio-temporal SSM fusion (the ChangeMamba mechanism): concatenate the t0
    and t1 token sequences and run a bidirectional selective scan so the state
    space carries information across the temporal boundary, then return the
    (t1 - t0) change feature."""

    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = _Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)

    def _bidir(self, t):
        return 0.5 * (self.mamba(t) + torch.flip(self.mamba(torch.flip(t, [1])), [1]))

    def forward(self, fa, fb):  # [B, C, H, W] x2
        B, C, H, W = fa.shape
        ta = fa.flatten(2).transpose(1, 2)
        tb = fb.flatten(2).transpose(1, 2)
        y = self._bidir(self.norm(torch.cat([ta, tb], 1)))  # [B, 2L, C]
        ya, yb = y[:, : H * W], y[:, H * W :]
        return (yb - ya).transpose(1, 2).reshape(B, C, H, W)


class ChangeMamba(nn.Module):
    """ChangeMamba-style (Chen et al. 2024): a Siamese visual-Mamba encoder
    (bidirectional state-space blocks) with spatio-temporal SSM fusion of the
    bitemporal features and a light all-conv decoder.  Reimplemented in pure
    PyTorch (mamba_ssm selective scan) and trained from scratch on the same
    split.  SSM blocks run on the 32/16/8 grids to stay tractable on one GPU."""

    def __init__(self, ch=3, dims=(64, 128, 256), dec=128):
        super().__init__()
        assert _Mamba is not None, "mamba_ssm is required for --arch changemamba"
        self.stem = nn.Sequential(
            nn.Conv2d(ch, dims[0], 7, stride=4, padding=3),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )  # 256->32
        self.b0 = _VSSBlock(dims[0])
        self.down1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], 3, stride=2, padding=1), nn.BatchNorm2d(dims[1]), nn.GELU()
        )  # 32->16
        self.b1 = _VSSBlock(dims[1])
        self.down2 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], 3, stride=2, padding=1), nn.BatchNorm2d(dims[2]), nn.GELU()
        )  # 16->8
        self.b2 = _VSSBlock(dims[2])
        self.f1, self.f2 = _STMamba(dims[1]), _STMamba(dims[2])  # spatio-temporal SSM fusion
        self.proj = nn.ModuleList([nn.Conv2d(d, dec, 1) for d in dims])
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * dec, dec, 1), nn.BatchNorm2d(dec), nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(dec, 1, 1)

    def _enc(self, x):
        x = self.stem(x)
        e0 = self.b0(x)
        e1 = self.b1(self.down1(e0))
        e2 = self.b2(self.down2(e1))
        return e0, e1, e2

    def forward(self, x):
        a, b = x[:, :3], x[:, 3:]
        a0, a1, a2 = self._enc(a)
        b0, b1, b2 = self._enc(b)
        d0 = torch.abs(a0 - b0)  # shallow: plain difference (large grid, kept cheap)
        d1, d2 = self.f1(a1, b1), self.f2(a2, b2)
        H0, W0 = d0.shape[2:]
        feats = [self.proj[0](d0)]
        for i, d in enumerate((d1, d2), start=1):
            feats.append(
                F.interpolate(self.proj[i](d), size=(H0, W0), mode="bilinear", align_corners=False)
            )
        x = self.fuse(torch.cat(feats, 1))
        x = self.out(x)
        return F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False).squeeze(1)


def dice_bce(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


def mask_to_response(mask: np.ndarray) -> str:
    """Connected components -> [x,y,x,y] boxes normalised to [0,100]."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area < 8:  # drop specks
            continue
        boxes.append(
            [
                round(x / HW * 100),
                round(y / HW * 100),
                round((x + w) / HW * 100),
                round((y + h) / HW * 100),
            ]
        )
    if not boxes:
        return "No changes detected."
    return ", ".join(f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]" for b in boxes) + "."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument(
        "--arch",
        default="fc_siam_diff",
        choices=["fc_siam_diff", "unet", "snunet", "tinycd", "changeformer", "changemamba"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--limit-eval", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds_name, task_name, eval_stem = TASKS[args.task]
    dev = "cuda"

    train_rows = [
        r
        for r in json.loads(Path(f"{DATA_BASE}/TEOChatlas/train/instruct.json").read_text())
        if r.get("dataset") == ds_name and r.get("task") == task_name
    ]
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    eval_rows = json.loads(Path(f"{DATA_BASE}/TEOChatlas/eval/{eval_stem}.json").read_text())
    if args.limit_eval:
        eval_rows = eval_rows[: args.limit_eval]
    print(
        f"[{args.task}] arch={args.arch} seed={args.seed} train={len(train_rows)} eval={len(eval_rows)}",
        flush=True,
    )

    model = {
        "fc_siam_diff": FCSiamDiff,
        "unet": UNet,
        "snunet": SNUNet,
        "tinycd": TinyCD,
        "changeformer": ChangeFormer,
        "changemamba": ChangeMamba,
    }[args.arch]().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tl = DataLoader(
        CDDataset(train_rows, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    model.train()
    for ep in range(args.epochs):
        tot = 0.0
        for pair, mask, _ in tl:
            pair, mask = pair.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)
            opt.zero_grad()
            loss = dice_bce(model(pair), mask)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"  epoch {ep} loss={tot / len(tl):.4f}", flush=True)

    # ---- eval -> predictions in TEOChat response schema
    model.eval()
    el = DataLoader(
        CDDataset(eval_rows, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    outputs = [None] * len(eval_rows)
    with torch.inference_mode():
        for pair, _, idx in el:
            prob = torch.sigmoid(model(pair.to(dev))).cpu().numpy()
            for p, i in zip(prob, idx.tolist()):
                r = eval_rows[i]
                resp = mask_to_response((p > 0.5).astype(np.uint8))
                out = {
                    "response": resp,
                    "ground_truth": r["conversations"][1]["value"],
                    "task": r["task"],
                }
                if r.get("polygon") is not None:
                    out["polygon"] = r["polygon"]
                outputs[i] = out

    out_subdir = args.out_dir / args.task
    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / OUT_NAME).write_text(json.dumps(outputs, indent=4), encoding="utf-8")
    print(f"[{args.task}] WROTE {out_subdir / OUT_NAME} ({len(outputs)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
