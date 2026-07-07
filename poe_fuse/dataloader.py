"""Dataset task registry for the three leaderboard targets.

Only ``s2_det``, ``xbd_dmg_cls`` and ``xbd_loc`` are in scope.

Earlier revisions of this file shipped a wide ``DatasetTaskSpec`` table
covering fMoW / UCMerced / QFabric / MP4 motion-vector caches; all of
those entries pointed at deleted modules (``model.remora``,
``MVCubeDataset``) and have been removed.  The image-pair pipeline lives
in ``poe_fuse.data.ImagePairDataset``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TEOCHATLAS_EVAL_DIR = Path("/data/TEOChatlas/eval")
DEFAULT_TEOCHATLAS_TRAIN_JSON = Path("/data/TEOChatlas/train/instruct.json")


@dataclass(frozen=True)
class DatasetTaskSpec:
    """Routing record for one of the three leaderboard tasks.

    Attributes:
        key:                stable identifier used throughout the codebase
                            (also the YAML config filename stem).
        dataset:            human-readable dataset name.
        task:               value of the official ``record['task']`` field
                            consumed by
                            :func:`teochat_eval.detection.detection_metrics`
                            *and* of ``record['task']`` in the official
                            TEOChatlas ``train/instruct.json``.
        metric:             key in ``detection_metrics`` return dict that
                            we treat as the marquee metric for the
                            leaderboard.  Always one of
                            ``{task}_f1`` / ``{task}_accuracy``.
        eval_dataset_name:  value passed as ``dataset_name`` to
                            ``detection_metrics`` -- must match the
                            upstream assertions inside that function.
        score_column:       column name in the wide leaderboard CSV.
        head_kind / label_kind: PoEFuse head/dataset routing tags.
        train_filter_dataset: value of ``record['dataset']`` in
                            ``train/instruct.json`` to filter to this task
                            (e.g. ``"S2Looking"`` or ``"xBD"``).  The
                            ``task`` field is also matched against
                            ``self.task``.
        sam3_default_prompt: short noun phrase fed to SAM 3 as the text
                            prompt for every sample of this task.  An empty
                            string tells the PoEFuse pipeline to fall back
                            to the sample's own ``question`` field.
    """

    key: str
    dataset: str
    task: str
    metric: str
    eval_dataset_name: str
    score_column: str
    head_kind: str
    label_kind: str
    eval_json: Path
    default_config: Path
    train_filter_dataset: str
    sam3_default_prompt: str
    # Eval scorer: "detection" routes through teochat detection_metrics;
    # "classification" uses exact-match accuracy (teochat
    # classification_metrics).
    eval_scorer: str = "detection"
    # Classification label vocabulary (see poe_fuse.data.CLS_VOCABS).
    cls_vocab: str = "xbd_damage"
    # Minimum #frames a record must have to be usable (CD needs 2; fMoW 1).
    min_frames: int = 2

    @property
    def eval_exists(self) -> bool:
        return self.eval_json.exists()


TASK_SPECS: tuple[DatasetTaskSpec, ...] = (
    DatasetTaskSpec(
        key="s2_det",
        dataset="S2Looking Det.",
        task="change_detection_detection",
        metric="change_detection_detection_f1",
        eval_dataset_name="s2_det",
        score_column="s2_det",
        head_kind="bbox_grid",
        label_kind="bbox",
        eval_json=DEFAULT_TEOCHATLAS_EVAL_DIR / "S2Looking_Change_Detection.json",
        default_config=Path("configs/poe_fuse/s2_det.yaml"),
        train_filter_dataset="S2Looking",
        sam3_default_prompt="building",
    ),
    DatasetTaskSpec(
        key="xbd_dmg_cls",
        dataset="xBD Dmg Cls.",
        task="change_detection_classification",
        metric="change_detection_classification_f1",
        eval_dataset_name="xbd_dmg_cls",
        score_column="xbd_dmg_cls",
        head_kind="classifier",
        label_kind="cls",
        eval_json=DEFAULT_TEOCHATLAS_EVAL_DIR / "xBD_Change_Detection_Classification.json",
        default_config=Path("configs/poe_fuse/xbd_dmg_cls.yaml"),
        train_filter_dataset="xBD",
        sam3_default_prompt="damaged building",
    ),
    DatasetTaskSpec(
        key="xbd_loc",
        dataset="xBD Loc.",
        task="change_detection_localization",
        metric="change_detection_localization_f1",
        eval_dataset_name="xbd_loc",
        score_column="xbd_loc",
        head_kind="bbox_grid",
        label_kind="bbox",
        eval_json=DEFAULT_TEOCHATLAS_EVAL_DIR / "xBD_Change_Detection_Localization.json",
        default_config=Path("configs/poe_fuse/xbd_loc.yaml"),
        train_filter_dataset="xBD",
        sam3_default_prompt="building",
    ),
)


def spec_by_key(key: str) -> DatasetTaskSpec:
    """Return the :class:`DatasetTaskSpec` whose ``key`` matches ``key``."""
    for spec in TASK_SPECS:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown dataset task key: {key}")


SCORE_COLUMNS: tuple[str, ...] = tuple(spec.score_column for spec in TASK_SPECS)


def task_specs(keys: Iterable[str] | None = None) -> list[DatasetTaskSpec]:
    if keys is None:
        return list(TASK_SPECS)
    wanted = list(keys)
    by_key = {spec.key: spec for spec in TASK_SPECS}
    missing = [k for k in wanted if k not in by_key]
    if missing:
        raise KeyError(f"unknown dataset task key(s): {', '.join(missing)}")
    return [by_key[k] for k in wanted]


def read_json_records(path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    if max_records == 0:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected JSON list at {path}; got {type(raw).__name__}")
    rows = [row for row in raw if isinstance(row, dict)]
    return rows[:max_records] if max_records is not None else rows


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(raw, dict):
                yield raw


__all__ = [
    "DatasetTaskSpec",
    "DEFAULT_TEOCHATLAS_EVAL_DIR",
    "DEFAULT_TEOCHATLAS_TRAIN_JSON",
    "SCORE_COLUMNS",
    "TASK_SPECS",
    "iter_jsonl",
    "read_json_records",
    "spec_by_key",
    "task_specs",
]
