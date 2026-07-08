# PoE-Fuse

Precision-weighted **Product-of-Experts (PoE) fusion** of frozen foundation-model
features for multi-task remote-sensing **change detection**.

Three frozen heterogeneous experts (a vision ViT, a segmentation model, and a
vision-language model) are projected to a shared dimension and resampled onto a
common `G×G` grid. Their per-cell features are treated as noisy Gaussian
observations of one latent scene and combined by the **inverse-variance
(BLUE / Gaussian-PoE) posterior mean**, using a *learned* per-cell precision.
Uniform summation and scalar gating are exact special cases. A single lightweight
shared trunk (linear projections + PoE fusion + a Mamba-3 mixer + per-task heads)
solves three change-detection tasks at once: change detection, building
localization, and damage classification.

## Install

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .            # core (train / evaluate on cached features)
uv pip install -e ".[features]"  # + extracting features from raw imagery
```

Installation requires an NVIDIA GPU with the CUDA toolkit: `mamba-ssm`
builds CUDA extensions at install time.

Extracting features from raw imagery additionally requires the three frozen
encoders. The vision ViT and segmentation model install from their upstream
repositories (not on PyPI); the vision-language encoder loads via `transformers`.
The segmentation model's package additionally requires `pycocotools`.

## Usage

The trunk trains on **cached, frozen expert features** (the experts are never
fine-tuned), so the typical flow is: cache features once, then train / evaluate.

```bash
# 1. Cache frozen features for a task (writes feature_cache/<task>/{train,eval})
python scripts/cache_features.py --config configs/s2_seg_head_poe.yaml --out-dir feature_cache/s2_det

# 2. Train the shared PoE-Fuse trunk on the three change-detection tasks
python scripts/train_multitask.py \
    --tasks-file configs/tasks_3_poe_focal.txt \
    --epochs 30 --seed 42 \
    --out results.json --ckpt checkpoint.pt

# 3. Evaluate a checkpoint
python scripts/eval_multitask.py --tasks-file configs/tasks_3_poe_focal.txt --checkpoint checkpoint.pt

# 4. Specialist change-detection baselines (from scratch)
python scripts/cd_baselines.py --task s2_det --arch fc_siam_diff --seed 0
```

`--grad-clip` (gradient clipping) and `--eval-only` (score a checkpoint without
training) are also available on `train_multitask.py`.

## Ablations

`configs/ablation/` holds the fusion-rule, fusion-site, and expert-subset
ablations. Each arm is a shared-trunk config triple plus a `tasks_<arm>.txt`
list, runnable with `train_multitask.py --tasks-file configs/ablation/tasks_<arm>.txt`.
The precision form is controlled by `FusionConfig` fields
(`poe_fuse`, `poe_static`, `poe_diff_corollary`, `poe_dense`, `poe_cls`) and the
active experts by `FusionConfig.active_experts`.

## Layout

```
poe_fuse/            model package (config, fusion, codec, backbones, heads, model, mixers, ...)
poe_fuse/metrics/    per-pixel F1 / classification metrics
poe_fuse/train.py    training + evaluation library
scripts/             entry points (train / eval / cache features / baselines)
configs/             experiment + ablation configs
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
