"""Vendored copy of TEOChat's ``videollava.eval`` package.

These files are byte-for-byte copies of the official TEOChat evaluation
code (https://github.com/ermongroup/TEOChat) at::

    TEOChat/videollava/eval/classification.py
    TEOChat/videollava/eval/detection.py

We vendor them so that:

* the project does not need the full ``videollava`` model stack (and its
  pinned ``transformers`` version) installed just to compute eval metrics,
* the metric computation matches the upstream eval bit-for-bit, removing
  any drift between our re-implementations and the paper-side numbers.

The only modification is in ``detection.py``: the absolute import
``from videollava.eval.classification import ...`` was rewritten as the
relative ``from .classification import ...`` so the module works inside
this package.
"""

from .classification import (
    classification_metrics,
    get_string_cleaner,
)
from .detection import (
    Evaluator,
    change_detection_classification,
    create_mask,
    detection_metrics,
    evaluate_masks,
    get_classes,
)

__all__ = [
    "Evaluator",
    "change_detection_classification",
    "classification_metrics",
    "create_mask",
    "detection_metrics",
    "evaluate_masks",
    "get_classes",
    "get_string_cleaner",
]
