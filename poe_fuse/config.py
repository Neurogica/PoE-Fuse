"""PoEFuse configuration dataclasses (trio-backbone + Mamba3 stack variant).

This rewrite drops the original DINOv2+SigLIP+HMSS pipeline.  The codec now
runs three frozen backbones (DINOv3 ViT, SAM3 image model, Gemma-4 LLM),
extracts the penultimate hidden representation from each, per-branch
linear-projects to ``d_s``, sequence-concats them, then runs a Mamba-3 stack
across the resulting token stream.

The three target tasks (s2_det / xbd_dmg_cls / xbd_loc) all
remain image-pair tasks: ``T=2`` images per sample produce two of these token
blocks which are concatenated along the sequence axis before the Mamba-3
stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TrioBackboneConfig:
    """Frozen-backbone configuration for the trio (Gemma-4, DINOv3, SAM3).

    All three backbones are loaded once at ``PoEFuse(cfg)`` build time and kept
    frozen for the whole training run.  ``image_size`` is shared so the codec
    feeds the same resized tensor to every backbone (each one then re-resizes
    internally as needed).
    """

    image_size: int = 224

    # DINOv3 ViT-7B/16 (Meta satellite pretrain by default).
    #
    # The Meta-native ``.pth`` checkpoint (``dinov3_weights``) is gated behind a
    # signed download URL and is not redistributed via the HF Hub, so by default
    # we load the equivalent ``transformers`` (``DINOv3ViTModel``) safetensors
    # snapshot from ``dinov3_model_dir`` instead.  Set ``dinov3_use_hf=False`` to
    # fall back to the original ``dinov3`` package + ``.pth`` path.
    dinov3_repo_dir: str = "models/dinov3_repo"
    dinov3_arch: str = "dinov3_vit7b16"
    dinov3_weights: str = "models/dinov3_vit7b16_pretrain_sat493m-a6675841.pth"
    dinov3_use_hf: bool = True
    dinov3_model_dir: str = "models/facebook/dinov3-vit7b16-pretrain-sat493m"
    dinov3_hidden_size: int = 4096
    dinov3_n_tokens: int = 196

    # SAM 3 image model (DETR-based detector).
    sam3_model_dir: str = "models/facebook/sam3"
    sam3_hidden_size: int = 256
    sam3_n_tokens: int = 200

    # Gemma-4 E4B-it multimodal LLM.
    gemma4_model_dir: str = "models/google/gemma-4-E4B-it"
    gemma4_hidden_size: int = 2560
    gemma4_n_image_tokens: int = 256
    gemma4_image_token_id: int = 258880

    # dtype the three backbones run their forward in.
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"

    # Whether DINOv3 / SAM3 / Gemma4 must be loaded eagerly (errors are
    # raised at construction time) or lazily (errors are deferred to the
    # first forward).  Eager is safer for distributed training; lazy is
    # useful when one of the heavy checkpoints is missing on a dev host.
    lazy_load: bool = False


@dataclass
class Mamba3StackConfig:
    """Mamba-3 token-mixer stack.

    Plugs directly into :class:`poe_fuse.mamba3.Mamba3` blocks.  We
    wrap each block in a pre-norm residual (``x + block(RMSNorm(x))``) so the
    same configuration matches the standard SSM language-modeling recipe.
    """

    d_model: int = 1024
    n_layers: int = 4
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    chunk_size: int = 64
    is_mimo: bool = False
    mimo_rank: int = 4
    is_outproj_norm: bool = False
    dropout: float = 0.0


@dataclass
class FusionConfig:
    """Cross-expert change fusion over the frozen trio.

    The frozen experts produce *heterogeneous* bi-temporal features (DINOv3
    geometry patches, SAM 3 detection queries, Gemma-4 semantic soft tokens).
    Rather than naively concatenating both time steps, this module turns the trio
    into a change detector by:

    1. (internal) forming per-expert bi-temporal *difference* tokens
       ``D_e = p_e(t_last) - p_e(t0)`` (+ optional ``|D_e|``);
    2. pooling each expert's change signal into a summary and running a small
       *cross-expert attention* so experts can corroborate / complement each
       other's change evidence;
    3. emitting (a) a learned per-expert **gate** that re-weights confident,
       agreed-upon change and suppresses expert-specific noise, and (b) a few
       global **agreement tokens** injected into the Mamba sequence.

    All of this is trainable and runs on the cached frozen features, so it is
    cheap.  Every component is individually ablatable; with all flags off the
    codec falls back to the original concat path.
    """

    enabled: bool = False
    diff_tokens: bool = True  # append per-expert signed difference tokens
    diff_abs: bool = True  # also append |difference| tokens
    cross_expert_gate: bool = True  # learned per-expert change gate
    agreement_tokens: bool = True  # inject cross-expert agreement summary tokens
    n_heads: int = 4
    add_segment_embeddings: bool = True  # mark t0 / t1 / diff / agreement roles

    # --- Spatial Cross-Expert Change-Agreement (the headline ACCV path) ----
    # When ``spatial_align`` is on, the codec swaps the global-pool path above
    # for :class:`SpatialChangeAgreementFusion`, which (2) resamples every
    # heterogeneous expert onto a *shared* ``align_grid x align_grid`` spatial
    # field with a learned cross-attention resampler, then (1) measures the
    # per-location agreement between the experts' change directions and uses it
    # to amplify corroborated change / suppress expert-specific pseudo-change.
    spatial_align: bool = False  # (2) resample experts onto a shared grid
    align_grid: int = 14  # G; produces G*G aligned tokens / expert / t
    align_heads: int = 8  # heads of the resampler cross-attention
    spatial_agreement: bool = True  # (1) per-location cross-expert agreement gate
    keep_change_tokens: bool = True  # emit the agreement-weighted change map tokens

    # --- Bi-temporal cross-alignment (novel: deformable temporal correspondence)
    # Before differencing, cross-attend each location of the t1 grid into the
    # *neighbourhood* of the t0 grid (and vice versa) so the change operator can
    # match spatially-displaced content (a moved/new building) rather than only
    # co-located pixels.  This makes the change token a learned correspondence
    # field instead of a naive frame subtraction.
    bitemporal_align: bool = False
    bitemporal_heads: int = 8

    # --- Unbalanced optimal-transport temporal correspondence (headline novelty)
    # Requires ``bitemporal_align=true``.  Replaces the single-MHA ``_temporal_match``
    # with an (approximately doubly-stochastic) entropic-OT plan on a learned ground
    # metric.  Emits a per-cell transport-cost residual ``tau`` + unmatched-mass
    # ("appeared/disappeared") signal that modulate the fused change field via a
    # warm-started ``cost_scale`` (init 0 => starts at the incumbent behaviour).
    #   sinkhorn_iters=0    => exactly row-softmax attention (incumbent matcher)
    #   unbalanced_rho=large => balanced OT (kills the appearance term; ablation)
    transport_align: bool = False
    sinkhorn_iters: int = 4
    sinkhorn_eps: float = 0.1  # entropic reg (learned, warm-started here)
    unbalanced_rho: float = 4.0  # KL marginal weight (learned, warm-started here)
    transport_cost_channel: bool = True  # fold tau/unmatched into the change field
    # --- Additive transport-evidence variants (v11; the v10b post-mortem fix) --
    # Both keep d_e = a1 - a0 untouched (no barycentric blur) and ADD an OT-derived
    # residual into the d_model change field with a live-gradient warm-start, so the
    # baseline is ~recovered at init but transport can actually be evaluated.
    transport_cost_only: bool = False  # (A) add tau (transport cost) + row-entropy H
    transport_marginal: bool = False  # (B) add bidirectional unbalanced mass deficits
    decouple_lam: bool = False  # learn Sinkhorn damping lam directly (needed for B)
    lam_init: float = 0.7  # initial lam when decoupled
    cost_alpha_init: float = 0.05  # (A) residual scale warm-start (NOT 0 -> live grad)
    # Warm-start of the correspondence blend and tau modulation.  Defaults keep
    # the proven naive-difference start (temporal_mix=-4 => sigmoid~=0.018,
    # cost_scale=0 => no-op).  Raise both to FORCE the transport path on from the
    # start (e.g. temporal_mix_init=0.0 => 0.5 blend) to actually exercise OT.
    temporal_mix_init: float = -4.0
    cost_scale_init: float = 0.0
    # --- v12 head-aware transport (one OT plan, two readouts) ------------------
    # transport_pool: enable the v12 path. Dense change_seg heads get a spatial
    # conv decoder of the 4 OT functionals added into change_grid (transport_dense
    # =true); the fragile classifier head instead receives a pooled 8-stat summary
    # threaded into ClassifierHead through its OWN zero-init gate -- so the
    # classifier token stream / change field is NEVER perturbed (the v11b collapse
    # fix). Set transport_dense=false on the classifier config.
    transport_pool: bool = False
    transport_dense: bool = True
    td_alpha_init: float = 0.1
    # --- PoE-Fuse: precision-weighted Product-of-Experts fusion (central novelty) -
    # Replaces the gated SUM-over-experts (at BOTH combine sites: fused_t for the
    # classifier path and fused_change for the dense path) with the Gaussian-PoE
    # posterior mean = inverse-variance (precision) weighted estimator (BLUE).  Each
    # expert gets a learned per-cell-per-channel diagonal precision Lambda_e; the
    # incumbent uniform sum (Lambda=I) and scalar cross_expert_gate (Lambda=g_e*I)
    # are exact special cases.  poe_scale_init=0 + zero-init last layer => recovers
    # the incumbent at init (logit-identical through the downstream norms), with a
    # live gradient (prec_scale multiplies a non-zero logprec MLP).  This is the one
    # lever that edits what BOTH heads read, so it can reach the mean-pool classifier.
    poe_fuse: bool = False
    poe_scale_init: float = 0.1
    # Ablation: freeze prec_scale=0 so Lambda_e = softplus(prec_bias_e) is a
    # learned but *input-independent* per-expert precision (no per-cell MLP).
    # Isolates the contribution of the dynamic per-cell-per-channel precision.
    poe_static: bool = False
    # B (difference-precision corollary): change-field precision = harmonic combine
    # of the two timesteps' precisions, Var(x1-x0)=S0+S1.  Default on; ablation
    # off => single-timestep precision Lambda(t1) for the change fusion.
    poe_diff_corollary: bool = True
    # Fusion-site ablation (contribution: PoE reaches BOTH combine sites).  Apply
    # the PoE combine only at the dense site (``fused_change``) and/or the
    # classifier site (``fused_t``); a disabled site falls back to the incumbent
    # gated sum.  Both default True == the headline model.  Isolating either shows
    # the marginal value of precision fusion at that site -- in particular the
    # mean-pool classifier path that OT-style change operators cannot reach.
    poe_dense: bool = True
    poe_cls: bool = True

    # Subset of frozen experts to use (ablation / strong baselines).  Default
    # uses all three; set to e.g. ``["dino"]`` for a single-expert baseline.
    active_experts: tuple[str, ...] = ("dino", "sam3", "gemma")

    def __post_init__(self) -> None:
        ae = self.active_experts
        if isinstance(ae, list):
            object.__setattr__(self, "active_experts", tuple(ae))


MixerKind = Literal["mamba3", "mamba3_bidir", "transformer", "gated_mlp"]
ScanOrder = Literal["segment_major", "location_major"]

# Param-matched default depths (~25-27M trainable at d_model=1024) so the
# sequence-mixer ablation compares kinds at a matched parameter budget.
MIXER_DEFAULT_LAYERS: dict[str, int] = {
    "mamba3": 4,
    "mamba3_bidir": 2,
    "transformer": 2,
    "gated_mlp": 4,
}


@dataclass
class MixerConfig:
    """Sequence-mixer ablation axis.

    ``kind`` selects the token mixer that consumes the fused token stream
    (param-matched alternatives to the default Mamba-3 stack).  ``scan_order``
    selects the serialization of the spatially-aligned bitemporal tokens fed
    to an SSM:

    * ``segment_major`` (default / legacy): ``[t0 grid][t1 grid][change grid]``
      -- temporal comparison must bridge G*G tokens of state.
    * ``location_major`` (Change-Anchored Scan): per grid location the
      ``(t0_i, t1_i, change_i)`` tokens are adjacent -- temporal comparison is
      local, spatial context accumulates along the raster scan.  Only defined
      when the fusion aligns all experts onto the shared grid
      (``fusion.spatial_align``), which is exactly the point: alignment
      *unlocks* structured linear-time scanning.

    Defaults reproduce the legacy architecture exactly (state-dict compatible).
    """

    kind: str = "mamba3"
    scan_order: str = "segment_major"
    n_layers: int = 0  # 0 -> param-matched default per kind (MIXER_DEFAULT_LAYERS)
    n_heads: int = 8  # transformer only
    ffn_mult: int = 4  # transformer only

    def __post_init__(self) -> None:
        if self.kind not in MIXER_DEFAULT_LAYERS:
            raise ValueError(
                f"mixer.kind must be one of {sorted(MIXER_DEFAULT_LAYERS)}; got {self.kind!r}"
            )
        if self.scan_order not in ("segment_major", "location_major"):
            raise ValueError(
                f"mixer.scan_order must be 'segment_major' or 'location_major'; "
                f"got {self.scan_order!r}"
            )
        if self.kind in ("transformer", "gated_mlp") and self.scan_order != "segment_major":
            # Order-invariant (attention) or no-mixing kinds: scan order is
            # meaningless, normalise to the default so configs stay honest.
            object.__setattr__(self, "scan_order", "segment_major")

    @property
    def resolved_layers(self) -> int:
        return self.n_layers if self.n_layers > 0 else MIXER_DEFAULT_LAYERS[self.kind]


HeadKind = Literal["classifier", "bbox_grid", "change_seg", "gemma_qa"]


@dataclass
class HeadConfig:
    """Task head configuration (identical surface to the previous codec)."""

    kind: HeadKind = "classifier"

    num_classes: int = 5
    pool: Literal["mean", "cls"] = "mean"
    # Classifier focal-CE: down-weight the easy majority (no-damage) that leaks
    # into minor. 0.0 = plain weighted CE (incumbent). Keeps inverse-freq weights.
    cls_focal_gamma: float = 0.0

    grid_side: int = 7
    bbox_lambda_box: float = 5.0
    bbox_lambda_obj: float = 1.0
    bbox_obj_threshold: float = 0.5
    bbox_focal_alpha: float = 0.25
    bbox_focal_gamma: float = 2.0

    # Dense change-segmentation head (``change_seg``): predicts a per-pixel
    # change mask, the native S2Looking task that the official pixel-F1 metric
    # actually expects (the bbox proxy saturates that metric).
    seg_out_size: int = 32  # R; head predicts an R x R change-mask grid
    seg_heads: int = 4  # attention heads of the mask decoder
    seg_num_layers: int = 2  # transformer-decoder layers of the mask head
    seg_lambda_dice: float = 1.0  # weight of the soft-Dice term (vs BCE)
    seg_pos_weight: float = 5.0  # BCE positive (change) weight; change is rare
    seg_threshold: float = 0.5  # sigmoid cut at eval time
    text_vocab_size: int = 103
    text_max_len: int = 96
    text_num_layers: int = 2
    text_num_heads: int = 4
    text_dropout: float = 0.0

    # GemmaQADecoderHead
    gemma_model_dir: str = "models/google/gemma-4-E4B-it"
    n_adapter_layers: int = 4
    adapter_heads: int = 8
    max_gen_len: int = 64


@dataclass
class PoEFuseConfig:
    """Top-level PoEFuse configuration (trio-backbone + Mamba3 stack)."""

    backbone: TrioBackboneConfig = field(default_factory=TrioBackboneConfig)
    mamba3: Mamba3StackConfig = field(default_factory=Mamba3StackConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    mixer: MixerConfig = field(default_factory=MixerConfig)

    def __post_init__(self) -> None:
        if self.mixer.scan_order == "location_major" and not (
            self.fusion.enabled and self.fusion.spatial_align
        ):
            raise ValueError(
                "mixer.scan_order='location_major' requires fusion.spatial_align=true: "
                "non-aligned expert tokens have no grid coordinates to anchor the scan"
            )

    @property
    def d_s(self) -> int:
        return self.mamba3.d_model

    @property
    def image_size(self) -> int:
        return self.backbone.image_size


__all__ = [
    "FusionConfig",
    "HeadConfig",
    "HeadKind",
    "MIXER_DEFAULT_LAYERS",
    "Mamba3StackConfig",
    "MixerConfig",
    "PoEFuseConfig",
    "TrioBackboneConfig",
]
