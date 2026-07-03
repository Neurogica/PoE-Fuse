"""Frozen backbones used by the trio-backbone PoEFuse codec.

Each class is a thin ``nn.Module`` wrapper that loads one frozen foundation
model, registers a forward hook on its second-to-last block / layer, and
returns the captured penultimate hidden representation as a single tensor
``(B, N_t, d_t)`` from :meth:`forward`.

* :class:`DinoV3PenultimateHook` -- Meta DINOv3 ViT-7B/16 satellite weights.
  ``(B, 196, 4096)`` at 224 input.
* :class:`Sam3PenultimateHook`   -- Meta SAM 3 image model, DETR decoder
  layer ``[-2]`` query embeddings.  ``(B, 200, 256)``.
* :class:`Gemma4PenultimateHook` -- Google Gemma-4-E4B-it ``language_model``
  hidden_states ``[-2]`` sliced to the per-image soft tokens.
  ``(B, 280, 2560)`` when the prompt contains exactly one image.

All three backbones are loaded in eval mode, every parameter is frozen, and
:meth:`train` is overridden so :meth:`PoEFuse.train()` cannot accidentally flip
them back into train mode.

These imports are deliberately inside the constructor bodies: the packages
(``dinov3``, ``sam3``, ``transformers>=5``) are huge and we want the rest of
the package to keep importing cleanly when only one of them is installed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TrioBackboneConfig

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"unsupported backbone dtype: {name!r}")
    return _DTYPE_MAP[name]


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


class _FrozenBackbone(nn.Module):
    """Shared ``train()`` override + parameter-freeze helper."""

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        for m in self.children():
            m.eval()
        return self


# -- DINOv3 ---------------------------------------------------------------


class DinoV3PenultimateHook(_FrozenBackbone):
    """DINOv3 ViT penultimate-block patch tokens.

    Two loading paths are supported:

    * ``cfg.dinov3_use_hf=True`` (default): load the ``transformers``
      ``DINOv3ViTModel`` safetensors snapshot from ``cfg.dinov3_model_dir`` and
      read ``output_hidden_states[-2]`` (the second-to-last block, no final
      norm).  This is the only path that works from HF Hub weights because the
      Meta-native ``.pth`` is gated behind a signed URL.
    * ``cfg.dinov3_use_hf=False``: load via the ``dinov3`` package using
      :func:`dinov3.hub.backbones.dinov3_vit7b16` + the native ``.pth`` and call
      ``get_intermediate_layers(block depth - 2)``.

    Either way the returned tensor strips the CLS + register/storage tokens so
    only the ``(B, N_patch, d)`` patch tokens are kept.
    """

    def __init__(self, cfg: TrioBackboneConfig):
        super().__init__()
        if cfg.dinov3_arch != "dinov3_vit7b16":
            raise NotImplementedError(
                f"DinoV3PenultimateHook only wires dinov3_vit7b16; got {cfg.dinov3_arch!r}"
            )
        self.dtype = _resolve_dtype(cfg.dtype)
        self.image_size = cfg.image_size
        self.expected_d = cfg.dinov3_hidden_size
        self.expected_n = cfg.dinov3_n_tokens
        self.use_hf = bool(getattr(cfg, "dinov3_use_hf", False))

        if self.use_hf:
            from transformers import AutoModel  # type: ignore

            model = AutoModel.from_pretrained(cfg.dinov3_model_dir, dtype=self.dtype)
            self.model = model
            # transformers emits ``num_hidden_layers + 1`` hidden states
            # (embeddings at [0]); ``[-2]`` is the second-to-last block output.
            self.penult_index = -2
        else:
            from dinov3.hub.backbones import dinov3_vit7b16  # type: ignore

            model = dinov3_vit7b16(pretrained=True, weights=cfg.dinov3_weights)
            # Cast the whole ViT to the codec's working dtype (defaults to bf16)
            # so its weights, biases and ``RopePositionEmbedding`` buffers all
            # agree with the inputs we feed it.
            self.model = model.to(self.dtype)
            # blocks[-2] -- the second-to-last transformer block.
            self.penult_index = model.n_blocks - 2
        _freeze(self)

    def forward(self, images: Tensor) -> Tensor:
        assert images.dim() == 4, f"images must be (B,3,H,W); got {tuple(images.shape)}"
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        # DINOv3 expects ImageNet-normalised tensors; callers (the codec) hand us
        # 0..1 images and we normalise here to match Meta's recipe.
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = ((images - mean) / std).to(self.dtype)

        with torch.inference_mode():
            if self.use_hf:
                outputs = self.model(pixel_values=x, output_hidden_states=True)
                hidden = outputs.hidden_states[self.penult_index]  # (B, 1+R+N_patch, d)
                # Patch tokens always come last regardless of CLS/register
                # placement, so slice the trailing ``expected_n`` positions.
                tokens = hidden[:, -self.expected_n :, :]
            else:
                outputs = self.model.get_intermediate_layers(
                    x,
                    n=[self.penult_index],
                    reshape=False,
                    return_class_token=False,
                    norm=False,
                )
                tokens = outputs[0]  # (B, N_patch, d_dinov3)

        if tokens.size(1) != self.expected_n:
            raise RuntimeError(
                f"DINOv3 produced {tokens.size(1)} patch tokens, expected {self.expected_n}; "
                "check image_size / patch_size"
            )
        if tokens.size(-1) != self.expected_d:
            raise RuntimeError(
                f"DINOv3 hidden_size {tokens.size(-1)} != configured {self.expected_d}"
            )
        return tokens.to(torch.float32)


# -- SAM3 -----------------------------------------------------------------


class Sam3PenultimateHook(_FrozenBackbone):
    """SAM 3 image-model DETR decoder penultimate query embeddings.

    We re-implement the bare minimum of :class:`sam3.model.sam3_image_processor.Sam3Processor`
    forward path (image backbone -> text backbone -> ``forward_grounding``)
    so we can run it in batch mode and capture the DETR decoder layer
    ``[-2]`` output via a forward hook.

    Returns ``(B, num_queries, 256)``.  SAM 3 uses 200 queries by default.
    """

    def __init__(self, cfg: TrioBackboneConfig):
        super().__init__()
        from sam3.model.data_misc import FindStage  # type: ignore
        from sam3.model_builder import build_sam3_image_model  # type: ignore

        ckpt_path = f"{cfg.sam3_model_dir}/sam3.pt"
        model = build_sam3_image_model(
            device="cpu",
            eval_mode=True,
            checkpoint_path=ckpt_path,
            load_from_HF=False,
            enable_segmentation=False,
            enable_inst_interactivity=False,
            compile=False,
        )
        self.dtype = _resolve_dtype(cfg.dtype)
        # SAM3's geometry encoder creates float32 helper tensors on the fly
        # (sinusoidal point embeddings, dummy prompts, ...) and broadcasts
        # them against parameter tensors.  Casting the whole model to bf16
        # therefore produces dtype-mismatch RuntimeErrors deep inside
        # ``forward_grounding``.  We keep SAM3 in float32 and cast its output
        # to ``self.dtype`` before the codec's projection.
        self.model = model
        self.FindStage = FindStage
        self.image_size = cfg.image_size
        self.expected_d = cfg.sam3_hidden_size
        self.expected_n = cfg.sam3_n_tokens
        # SAM3 internal resolution (used by Sam3Processor's transform).
        self.sam3_resolution = 1008

        # Register hook on the DETR decoder's penultimate layer.
        decoder = model.transformer.decoder
        if not hasattr(decoder, "layers") or len(decoder.layers) < 2:
            raise RuntimeError("SAM3 transformer decoder is missing a ``layers`` ModuleList")
        self._hooked_output: Tensor | None = None

        def _capture(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self._hooked_output = tensor

        decoder.layers[-2].register_forward_hook(_capture)
        _freeze(self)

    def _preprocess(self, images: Tensor) -> Tensor:
        # SAM3 uses 0.5/0.5 mean/std at resolution=1008 with float32 inputs.
        side = self.sam3_resolution
        x = F.interpolate(images, size=(side, side), mode="bilinear", align_corners=False)
        mean = x.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = x.new_tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, images: Tensor, prompts: list[str]) -> Tensor:
        assert images.dim() == 4, f"images must be (B,3,H,W); got {tuple(images.shape)}"
        B = images.size(0)
        if len(prompts) != B:
            raise ValueError(
                f"SAM3 expects one prompt per sample; got {len(prompts)} for batch {B}"
            )
        device = images.device
        # SAM3's parameters stay in fp32, but ``sam3.perflib.fused.addmm_act``
        # unconditionally casts MLP inputs/weights to bf16 mid-block.  We run
        # the whole forward under bf16 autocast so subsequent fp32-weighted
        # linears (e.g. ``Mlp.fc2``) see consistent dtypes.
        x = self._preprocess(images).to(torch.float32)

        outputs_per_sample: list[Tensor] = []
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.inference_mode(), torch.autocast(amp_device, dtype=torch.bfloat16):
            for i in range(B):
                self._hooked_output = None
                backbone_out = {"img_batch_all_stages": x[i : i + 1]}
                backbone_out.update(self.model.backbone.forward_image(x[i : i + 1]))
                text_outputs = self.model.backbone.forward_text([prompts[i]], device=device)
                backbone_out.update(text_outputs)
                find_input = self.FindStage(
                    img_ids=torch.tensor([0], device=device, dtype=torch.long),
                    text_ids=torch.tensor([0], device=device, dtype=torch.long),
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )
                geometric_prompt = self.model._get_dummy_prompt()
                _ = self.model.forward_grounding(
                    backbone_out=backbone_out,
                    find_input=find_input,
                    find_target=None,
                    geometric_prompt=geometric_prompt,
                )
                if self._hooked_output is None:
                    raise RuntimeError("SAM3 decoder forward hook did not fire")
                hooked = self._hooked_output
                # Layer output is typically ``(num_queries, 1, d_model)`` (seq-first
                # batch=1) or ``(1, num_queries, d_model)``; normalise to
                # ``(num_queries, d_model)``.
                if hooked.dim() == 3 and hooked.size(1) == 1:
                    hooked = hooked.squeeze(1)
                elif hooked.dim() == 3 and hooked.size(0) == 1:
                    hooked = hooked.squeeze(0)
                else:
                    raise RuntimeError(
                        f"unexpected SAM3 decoder layer output shape: {tuple(hooked.shape)}"
                    )
                outputs_per_sample.append(hooked.unsqueeze(0))

        out = torch.cat(outputs_per_sample, dim=0)  # (B, num_queries, d_model)
        if out.size(1) != self.expected_n or out.size(-1) != self.expected_d:
            raise RuntimeError(
                f"SAM3 penultimate output shape {tuple(out.shape)} does not match "
                f"configured ({self.expected_n}, {self.expected_d})"
            )
        return out.to(torch.float32)


# -- Gemma 4 --------------------------------------------------------------


class Gemma4PenultimateHook(_FrozenBackbone):
    """Gemma-4-E4B-it LM penultimate hidden_state (image-soft-token slice).

    We feed the LLM ``image + question_text`` via
    :class:`transformers.AutoProcessor`, request ``output_hidden_states=True``
    for a single forward pass, and slice the resulting ``hidden_states[-2]``
    down to the positions whose ``input_ids`` equal
    :attr:`TrioBackboneConfig.gemma4_image_token_id`.  These positions are
    the per-image soft tokens (``vision_soft_tokens_per_image=280``).
    """

    def __init__(self, cfg: TrioBackboneConfig):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

        self.image_token_id = cfg.gemma4_image_token_id
        self.expected_d = cfg.gemma4_hidden_size
        self.expected_n = cfg.gemma4_n_image_tokens
        self.dtype = _resolve_dtype(cfg.dtype)

        self.processor = AutoProcessor.from_pretrained(cfg.gemma4_model_dir)
        model = AutoModelForImageTextToText.from_pretrained(
            cfg.gemma4_model_dir,
            dtype=self.dtype,
        )
        self.model = model
        _freeze(self)

    def _build_messages(self, image: Tensor, question: str) -> list[dict]:
        # ``image`` is (3,H,W) float32 in 0..1; convert to a PIL image so the
        # processor's image_processor picks the right resize / normalise path.
        from torchvision.transforms.functional import to_pil_image  # type: ignore

        pil = to_pil_image(image.clamp(0.0, 1.0))
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": question or "Describe this image."},
                ],
            }
        ]

    def forward(self, images: Tensor, questions: list[str]) -> Tensor:
        assert images.dim() == 4, f"images must be (B,3,H,W); got {tuple(images.shape)}"
        B = images.size(0)
        if len(questions) != B:
            raise ValueError(
                f"Gemma4 expects one question per sample; got {len(questions)} for batch {B}"
            )

        outputs_per_sample: list[Tensor] = []
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            for i in range(B):
                messages = self._build_messages(images[i].detach().cpu(), questions[i])
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
                out = self.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                # ``hidden_states`` is a tuple of (num_layers + 1) tensors:
                # the input embeddings followed by one entry per LM block.
                # ``[-1]`` is the final block output; ``[-2]`` is the
                # penultimate block (= "final layer 2 below the last").
                hidden = out.hidden_states[-2]
                ids = inputs["input_ids"][0]
                image_positions = (ids == self.image_token_id).nonzero(as_tuple=False).flatten()
                if image_positions.numel() != self.expected_n:
                    raise RuntimeError(
                        f"Gemma4 produced {image_positions.numel()} image soft tokens, "
                        f"expected {self.expected_n}; check the chat template / image token id"
                    )
                feat = hidden[0, image_positions, :]
                if feat.size(-1) != self.expected_d:
                    raise RuntimeError(
                        f"Gemma4 hidden_size {feat.size(-1)} != configured {self.expected_d}"
                    )
                outputs_per_sample.append(feat.unsqueeze(0))

        return torch.cat(outputs_per_sample, dim=0).to(torch.float32)


__all__ = [
    "DinoV3PenultimateHook",
    "Gemma4PenultimateHook",
    "Sam3PenultimateHook",
]
