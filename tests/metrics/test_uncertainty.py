"""Tests for `src/metrics/uncertainty.py`.

The load-bearing property here is *agreement*: these functions are the write-side
implementation of quantities the read side already computes its own way
(`src.metrics.io.shannon_entropy_nats`, `src.metrics.dempster_shafer_uncertainity`).
If they drift, a run's written column and the same run's derived column silently
disagree, which is worse than either being wrong on its own -- so those two
equivalences are asserted directly, following
`tests/metrics/test_registry.py::test_ood_auroc_matches_calculate_ood_metrics_exactly`.
"""

import math

import numpy as np
import pytest
import torch

from src.metrics.dempster_shafer_uncertainity import DempsterShaferUncertainty
from src.metrics.io import shannon_entropy_nats
from src.metrics.uncertainty import (
    GP_PREDICTIVE_VARIANCE,
    MC_LOGIT_STD,
    MEMBER_LOGIT_VARIANCE,
    UNKNOWN_UNCERTAINTY,
    confidence_margin,
    decompose_member_uncertainty,
    dempster_shafer,
    infer_uncertainty_kind,
    predictive_entropy,
)


# --- predictive_entropy ---


@pytest.mark.parametrize("num_classes", [2, 3, 10])
def test_uniform_probs_have_maximal_entropy(num_classes):
    probs = torch.full((1, num_classes), 1.0 / num_classes, dtype=torch.float64)
    assert predictive_entropy(probs).item() == pytest.approx(math.log(num_classes), abs=1e-9)


@pytest.mark.parametrize("num_classes", [2, 3, 10])
def test_float32_probs_stay_accurate_to_float32_precision(num_classes):
    """The real write path hands in float32 activations. The function computes in
    float64, so the only error left is the input's own representation of `1/C` -- ~1e-8,
    not the ~1e-16 the float64 path gives."""
    probs = torch.full((1, num_classes), 1.0 / num_classes, dtype=torch.float32)
    assert predictive_entropy(probs).item() == pytest.approx(math.log(num_classes), abs=1e-7)


def test_one_hot_probs_have_zero_entropy():
    probs = torch.zeros(1, 4)
    probs[0, 2] = 1.0
    # Exactly 0, not -1e-12: the clamp exists so a published column never goes negative.
    assert predictive_entropy(probs).item() == 0.0


def test_predictive_entropy_matches_the_read_side_implementation():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(16, 6), dim=1)
    expected = np.array([shannon_entropy_nats(row) for row in probs.tolist()])
    assert predictive_entropy(probs).numpy() == pytest.approx(expected, abs=1e-12)


def test_predictive_entropy_is_one_value_per_row():
    assert predictive_entropy(torch.softmax(torch.randn(5, 3), dim=1)).shape == (5,)


# --- confidence_margin ---


def test_margin_is_top1_minus_top2():
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    assert confidence_margin(probs).tolist() == pytest.approx([0.5, 0.05])


def test_confident_row_has_a_larger_margin_than_an_ambiguous_one():
    confident = torch.tensor([[0.98, 0.01, 0.01]])
    ambiguous = torch.tensor([[0.34, 0.33, 0.33]])
    assert confidence_margin(confident).item() > confidence_margin(ambiguous).item()


def test_margin_of_a_single_class_is_the_probability_itself():
    assert confidence_margin(torch.ones(2, 1)).tolist() == pytest.approx([1.0, 1.0])


# --- dempster_shafer ---


def test_dempster_shafer_matches_the_canonical_numpy_implementation():
    torch.manual_seed(1)
    logits = torch.randn(12, 5, dtype=torch.float64)
    expected = DempsterShaferUncertainty(logits.numpy())
    assert dempster_shafer(logits).numpy() == pytest.approx(expected, abs=1e-12)


def test_dempster_shafer_is_in_the_unit_interval():
    values = dempster_shafer(torch.randn(32, 4) * 5)
    assert values.min().item() >= 0.0 and values.max().item() <= 1.0


def test_more_evidence_means_less_dempster_shafer_uncertainty():
    weak = dempster_shafer(torch.zeros(1, 3)).item()
    strong = dempster_shafer(torch.tensor([[8.0, 0.0, 0.0]])).item()
    assert strong < weak


# --- decompose_member_uncertainty ---


def test_agreeing_members_have_zero_mutual_information():
    single = torch.randn(1, 7, 4)
    stack = single.repeat(5, 1, 1)  # every member identical

    decomposed = decompose_member_uncertainty(stack)

    assert decomposed.epistemic.abs().max().item() == pytest.approx(0.0, abs=1e-12)
    # With no disagreement all predictive uncertainty is aleatoric.
    assert decomposed.total.numpy() == pytest.approx(decomposed.aleatoric.numpy(), abs=1e-12)


def test_disagreeing_members_have_positive_mutual_information():
    # Two members confidently predicting opposite classes.
    stack = torch.tensor([[[10.0, -10.0]], [[-10.0, 10.0]]])
    decomposed = decompose_member_uncertainty(stack)

    assert decomposed.epistemic.item() > 0.5
    assert decomposed.total.item() > decomposed.aleatoric.item()


def test_decomposition_is_additive_and_non_negative():
    torch.manual_seed(2)
    decomposed = decompose_member_uncertainty(torch.randn(4, 9, 3) * 3)

    assert decomposed.total.numpy() == pytest.approx(
        (decomposed.aleatoric + decomposed.epistemic).numpy(), abs=1e-12
    )
    assert decomposed.epistemic.min().item() >= 0.0
    assert decomposed.aleatoric.min().item() >= 0.0


def test_decomposition_returns_one_value_per_sample_not_per_member():
    decomposed = decompose_member_uncertainty(torch.randn(5, 8, 3))
    assert decomposed.total.shape == (8,)
    assert decomposed.aleatoric.shape == (8,)
    assert decomposed.epistemic.shape == (8,)


@pytest.mark.parametrize(
    "shape",
    [(4, 3), (2, 3, 4, 5)],
    ids=["two-dim", "four-dim"],
)
def test_decomposition_rejects_anything_that_is_not_a_member_stack(shape):
    with pytest.raises(ValueError, match=r"\[M, B, C\]"):
        decompose_member_uncertainty(torch.randn(*shape))


# --- infer_uncertainty_kind ---


@pytest.mark.parametrize(
    "raw_logits, member_logits, mc_dropout, expected, case_id",
    [
        (torch.ones(2, 3), None, False, GP_PREDICTIVE_VARIANCE, "sngp-has-raw-logits"),
        (None, torch.ones(2, 2, 3), False, MEMBER_LOGIT_VARIANCE, "ensemble-has-members"),
        (None, torch.ones(2, 2, 3), True, MC_LOGIT_STD, "mc-dropout-wins-over-members"),
        (None, None, True, MC_LOGIT_STD, "mc-dropout-without-stack"),
        (None, None, False, UNKNOWN_UNCERTAINTY, "variance-of-unknown-origin"),
        # An ensemble *of SNGP members* reports member disagreement, not GP variance:
        # DeepEnsemble.forward discards its members' variances.
        (torch.ones(2, 3), torch.ones(2, 2, 3), False, MEMBER_LOGIT_VARIANCE, "sngp-ensemble"),
    ],
    ids=[
        "sngp-has-raw-logits",
        "ensemble-has-members",
        "mc-dropout-wins-over-members",
        "mc-dropout-without-stack",
        "variance-of-unknown-origin",
        "sngp-ensemble",
    ],
)
def test_uncertainty_kind_is_inferred_structurally(raw_logits, member_logits, mc_dropout, expected, case_id):
    kind = infer_uncertainty_kind(
        variance=torch.ones(2), raw_logits=raw_logits, member_logits=member_logits, mc_dropout=mc_dropout
    )
    assert kind == expected


def test_no_variance_means_no_kind():
    # A plain Baseline: there is no uncertainty column to label.
    assert (
        infer_uncertainty_kind(variance=None, raw_logits=None, member_logits=None, mc_dropout=False) is None
    )
