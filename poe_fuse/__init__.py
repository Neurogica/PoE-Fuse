"""PoEFuse public API (trio-backbone variant).

Image-pair pipeline (T=2 by default):

    images (B, T, 3, H, W) + questions + sam3_prompts
        -> PoEFuseCodec  (DINOv3 + SAM3 + Gemma4 penultimate hooks
                        + per-branch Linear -> Mamba3 stack)
        -> PoEFuse       (codec + task head)

Heads cover the three target tasks: ClassifierHead (xbd_dmg_cls),
BBoxGridHead (s2_det / xbd_loc).
"""

from .config import (
    FusionConfig,
    HeadConfig,
    Mamba3StackConfig,
    MixerConfig,
    PoEFuseConfig,
    TrioBackboneConfig,
)
from .data import (
    CLS_VOCABS,
    DEFAULT_IMAGE_BASE_CANDIDATES,
    FMOW_SCENE_LABELS,
    XBD_DAMAGE_LABELS,
    ImagePairDataset,
    format_bbox_list,
    sanitize_bbox_response,
)
from .feature_cache import (
    CachedTrioDataset,
    cached_collate,
    features_to_device,
    write_split_cache,
)
from .fusion import CrossExpertChangeFusion, SpatialChangeAgreementFusion
from .heads import ChangeSegHead, ReferringSegHead
from .model import PoEFuse

__all__ = [
    "CLS_VOCABS",
    "CachedTrioDataset",
    "ChangeSegHead",
    "ReferringSegHead",
    "CrossExpertChangeFusion",
    "DEFAULT_IMAGE_BASE_CANDIDATES",
    "FMOW_SCENE_LABELS",
    "FusionConfig",
    "SpatialChangeAgreementFusion",
    "HeadConfig",
    "ImagePairDataset",
    "Mamba3StackConfig",
    "MixerConfig",
    "PoEFuse",
    "PoEFuseConfig",
    "TrioBackboneConfig",
    "XBD_DAMAGE_LABELS",
    "cached_collate",
    "features_to_device",
    "format_bbox_list",
    "sanitize_bbox_response",
    "write_split_cache",
]
