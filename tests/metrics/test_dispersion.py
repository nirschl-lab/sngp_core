"""Tests for `src/metrics/dispersion.py`.

Two things matter beyond the arithmetic: the means must reproduce the metric values
`metrics.json` already reported (otherwise adding `_std` would silently move `nll`),
and `metrics.json` must stay JSON-round-trippable -- which is why a one-sample run
yields `0.0` rather than the NaN `torch.std` returns.
"""

import json
import math

import pytest
import torch

from src.metrics.brier import brier_score
from src.metrics.dispersion import (
    mean_std_sem,
    per_sample_brier,
    per_sample_correct,
    per_sample_nll,
    summarize_metric,
)


# --- mean_std_sem ---


def test_mean_std_and_sem_of_a_known_sample():
    values = torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    summary = mean_std_sem(values)

    assert summary.mean == pytest.approx(5.0)
    # Sample std (ddof=1), not the population std of 2.0.
    assert summary.std == pytest.approx(math.sqrt(32.0 / 7.0))
    assert summary.sem == pytest.approx(summary.std / math.sqrt(8))
    assert summary.n == 8


def test_sem_shrinks_with_sample_size_while_std_does_not():
    torch.manual_seed(0)
    small = mean_std_sem(torch.randn(100))
    large = mean_std_sem(torch.randn(10_000))

    assert large.sem < small.sem
    # Both sample the same distribution, so std stays ~1 regardless of n.
    assert large.std == pytest.approx(1.0, abs=0.1)


def test_identical_values_have_zero_spread():
    summary = mean_std_sem(torch.full((25,), 3.5))
    assert (summary.std, summary.sem) == (0.0, 0.0)


def test_a_single_sample_gives_zero_not_nan():
    summary = mean_std_sem(torch.tensor([1.5]))

    assert summary == (1.5, 0.0, 0.0, 1)
    # The reason it must not be NaN: metrics.json has to survive a JSON round-trip.
    assert json.loads(json.dumps({"m_std": summary.std}))["m_std"] == 0.0


def test_empty_input_is_an_error_not_a_nan():
    with pytest.raises(ValueError, match="at least one value"):
        mean_std_sem(torch.tensor([]))


def test_shape_is_flattened_before_summarizing():
    assert mean_std_sem(torch.ones(4, 1)).n == 4


# --- per-sample metrics ---


def test_per_sample_nll_is_the_negative_log_of_the_true_class_probability():
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    targets = torch.tensor([0, 1])

    values = per_sample_nll(probs, targets)

    assert values.tolist() == pytest.approx([-math.log(0.9), -math.log(0.8)], abs=1e-6)


def test_per_sample_nll_mean_reproduces_the_existing_metric():
    """`infer.py::_finalize` used to compute NLL with a mean-reduced `nll_loss` and the
    same 1e-8 clamp; adding dispersion must not move the reported `nll`."""
    torch.manual_seed(3)
    probs = torch.softmax(torch.randn(64, 5), dim=1)
    targets = torch.randint(0, 5, (64,))

    expected = torch.nn.functional.nll_loss(torch.log(probs + 1e-8), targets)

    assert per_sample_nll(probs, targets).mean().item() == pytest.approx(expected.item(), abs=1e-6)


def test_per_sample_brier_mean_reproduces_brier_score():
    torch.manual_seed(4)
    probs = torch.softmax(torch.randn(32, 4), dim=1)
    targets = torch.randint(0, 4, (32,))

    mean = per_sample_brier(probs, targets, num_classes=4).mean().item()

    assert mean == pytest.approx(brier_score(probs, targets, num_classes=4), abs=1e-6)


def test_per_sample_brier_bounds():
    perfect = torch.tensor([[1.0, 0.0]])
    worst = torch.tensor([[0.0, 1.0]])
    targets = torch.tensor([0])

    assert per_sample_brier(perfect, targets, num_classes=2).item() == pytest.approx(0.0)
    assert per_sample_brier(worst, targets, num_classes=2).item() == pytest.approx(2.0)


def test_per_sample_correct_mean_is_accuracy():
    preds = torch.tensor([0, 1, 2, 3])
    targets = torch.tensor([0, 1, 9, 9])

    values = per_sample_correct(preds, targets)

    assert values.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert values.mean().item() == pytest.approx(0.5)


# --- summarize_metric ---


def test_summarize_metric_emits_flat_sibling_keys():
    """Flat, not nested: `artifact_quantification._load_nll_brier` indexes
    `metrics.json["artifact.nll"]` directly and must keep working."""
    summary = summarize_metric("nll", torch.tensor([1.0, 2.0, 3.0]))

    assert set(summary) == {"nll", "nll_std", "nll_sem", "n_samples"}
    assert summary["nll"] == pytest.approx(2.0)
    assert summary["n_samples"] == 3


def test_summarize_metric_output_is_json_serializable():
    summary = summarize_metric("brier", torch.rand(10))
    assert json.loads(json.dumps(summary)) == pytest.approx(summary)
