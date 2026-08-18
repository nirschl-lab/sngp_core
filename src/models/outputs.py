"""Uniform return type for every net's `forward()`.

Every net in `src/models/{baseline,sngp,ensemble}/` returns a `ModelOutput` so callers
(LightningModules, `src/inference/infer.py`) never need to branch on model family to know
whether `forward()` returned a bare tensor or a family-specific tuple.
"""
from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass(frozen=True)
class ModelOutput:
    logits: Tensor  # always what loss/argmax use
    raw_logits: Optional[Tensor] = None  # SNGP: pre-mean-field-correction logits
    variance: Optional[Tensor] = None  # SNGP predictive variance / ensemble variance
    features: Optional[Tensor] = None  # penultimate features, when requested
    member_logits: Optional[Tensor] = None  # ensemble: per-member logits, [M, B, C]
