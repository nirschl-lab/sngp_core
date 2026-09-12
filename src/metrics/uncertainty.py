"""Per-sample uncertainty scalars, computed once at write time for every family.

The point of this module is comparability. `ModelOutput.variance` -- the `uncertainty`
column -- is *not* comparable across model families: it is a GP latent variance for
SNGP, a mean per-class logit **variance** for Deep Ensembles, and a mean per-class
logit **standard deviation** for MC-Dropout, while a plain Baseline has none at all.
Those are three different units and one absence, so ranking families by `uncertainty`
is meaningless. The scalars here are instead defined on quantities every family emits
(`logits`/`probs`, and a member stack where one exists), so they *are* comparable --
see `.claude/skills/metrics/references/csv_schema.md`.

Everything is torch-native and batched, because both callers hold tensors already:
`src/inference/records.py` (the `predictions.csv` write path) and
`src/callbacks/test_artifacts_callback.py` (the `trainer.test()` write path). Keeping
one implementation here is what stops those two writers from drifting the way their
column names already did.

These deliberately mirror the numpy read-side helpers rather than replacing them:
`src/metrics/io.py`'s `shannon_entropy_nats`/`normalized_entropy` still derive entropy
on *read* for runs written before `predictions_csv_schema: 3`, which have no entropy
column to read. `predictive_entropy` here matches `shannon_entropy_nats` (same `1e-12`
epsilon, same nats units) so the written column and the derived one agree for runs that
have both.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
from torch import Tensor

# Matches `src.metrics.io.shannon_entropy_nats`, so a written `predictive_entropy`
# column and io.py's read-derived `entropy_nats` agree to floating-point noise.
EPS = 1e-12

# `uncertainty_kind` values. The `uncertainty` column carries one of these units; the
# column is written next to it so a frame concatenated across families stays
# interpretable (see `infer_uncertainty_kind`).
GP_PREDICTIVE_VARIANCE = "gp_predictive_variance"
MEMBER_LOGIT_VARIANCE = "member_logit_variance"
MC_LOGIT_STD = "mc_logit_std"
UNKNOWN_UNCERTAINTY = "unknown"


class MemberUncertainty(NamedTuple):
    """The Depeweg et al. (2018) decomposition of predictive uncertainty, in nats.

    `total` is the entropy of the mean predictive distribution, `aleatoric` the mean of
    the members' individual entropies, and `epistemic` their difference -- the mutual
    information between the prediction and the model (BALD, Houlsby et al. 2011).
    Each is `[B]`.
    """

    total: Tensor
    aleatoric: Tensor
    epistemic: Tensor


def predictive_entropy(probs: Tensor) -> Tensor:
    """Shannon entropy of each row of `probs`, in nats. `[B, C] -> [B]`.

    Computed from probabilities rather than logits because the two disagree for
    MC-Dropout, where `class_probs` is the mean of per-pass softmax while
    `class_logits` is the mean of per-pass logits -- the former is the actual
    predictive distribution, so it is the one whose entropy is meaningful.

    Clamped at 0: the `EPS` inside the log makes a perfectly one-hot row come out at
    `-1e-12` rather than exactly 0, and a negative entropy in a published column is
    worth more confusion than the 1e-12 it costs to agree with
    `io.shannon_entropy_nats` exactly there.
    """
    p = probs.to(torch.float64)
    return (-(p * (p + EPS).log()).sum(dim=-1)).clamp(min=0.0)


def confidence_margin(probs: Tensor) -> Tensor:
    """Top-1 minus top-2 probability. `[B, C] -> [B]`.

    Small margin means the model nearly tied two classes -- an uncertainty signal that
    max-probability alone misses, since a confidently-wrong 0.9/0.09 row and an
    ambiguous 0.9/0.9-impossible row are not distinguishable by the max alone. Returns
    the top probability itself when there is only one class.
    """
    p = probs.to(torch.float64)
    if p.shape[-1] < 2:
        return p.squeeze(-1) if p.shape[-1] == 1 else torch.zeros(p.shape[0], dtype=p.dtype, device=p.device)
    top2 = p.topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def dempster_shafer(logits: Tensor) -> Tensor:
    """Dempster-Shafer uncertainty, `K / (K + sum_c exp(logit_c))`. `[B, C] -> [B]`.

    In `[0, 1]`, larger meaning less total evidence and so more uncertainty. Requires
    *logits*, not probabilities -- softmax is shift-invariant, so the total evidence
    mass this measures is destroyed by normalization, which is why `class_logits` is
    always persisted.

    Computed in float64 to match `src.metrics.dempster_shafer_uncertainity`'s numpy
    implementation (the canonical one, used by the `dempster_shafer` registered
    metric), so the written column reproduces that metric's value.
    """
    z = logits.to(torch.float64)
    num_classes = z.shape[-1]
    return num_classes / (z.exp().sum(dim=-1) + num_classes)


def decompose_member_uncertainty(member_logits: Tensor) -> MemberUncertainty:
    """Split predictive uncertainty into aleatoric and epistemic parts. `[M, B, C]`.

    `member_logits` is a stack of ensemble members or MC-Dropout passes -- the same
    shape convention either way, and the decomposition is identical for both: both are
    samples from an approximate posterior over models.

    The mean predictive distribution here is the mean of per-member **softmax**, which
    is the correct predictive distribution for a model average. Note this is *not*
    `softmax(ModelOutput.logits)` for a Deep Ensemble, whose `.logits` is the mean of
    per-member *logits* (`src/models/ensemble/deep_ensemble_model.py`) -- a
    geometric-style pooling. So `total` can differ slightly from the
    `predictive_entropy` column on the same row; that is a real property of the two
    poolings, not an inconsistency to reconcile.

    `epistemic` is clamped at 0: it is non-negative analytically (Jensen), but can go
    infinitesimally negative in floating point when every member agrees exactly.
    """
    if member_logits.dim() != 3:
        raise ValueError(f"member_logits must be [M, B, C]; got shape {tuple(member_logits.shape)}")

    probs_m = torch.softmax(member_logits.to(torch.float64), dim=-1)  # [M, B, C]
    mean_probs = probs_m.mean(dim=0)  # [B, C]

    total = (-(mean_probs * (mean_probs + EPS).log()).sum(dim=-1)).clamp(min=0.0)
    aleatoric = (-(probs_m * (probs_m + EPS).log()).sum(dim=-1)).mean(dim=0).clamp(min=0.0)
    return MemberUncertainty(total=total, aleatoric=aleatoric, epistemic=(total - aleatoric).clamp(min=0.0))


def infer_uncertainty_kind(
    *,
    variance: Optional[Tensor],
    raw_logits: Optional[Tensor],
    member_logits: Optional[Tensor],
    mc_dropout: bool,
) -> Optional[str]:
    """Name the unit of the `uncertainty` column, or `None` when there is no variance.

    Inferred structurally rather than from the net's registry name, so an
    SNGP/ensemble *variant* registered under a new key is labelled correctly without
    this module having to know about it. The three cases are mutually exclusive in
    practice: MC-Dropout never produces a `ModelOutput`, a Deep Ensemble is the only
    family with `member_logits`, and SNGP the only one with `raw_logits`.

    A Deep Ensemble *of SNGP members* lands on `MEMBER_LOGIT_VARIANCE`, correctly --
    `DeepEnsemble.forward` discards its members' GP variances and reports member
    disagreement instead.
    """
    if variance is None:
        return None
    if mc_dropout:
        return MC_LOGIT_STD
    if member_logits is not None:
        return MEMBER_LOGIT_VARIANCE
    if raw_logits is not None:
        return GP_PREDICTIVE_VARIANCE
    return UNKNOWN_UNCERTAINTY
