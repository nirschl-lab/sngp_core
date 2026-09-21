"""Smoke test for src/paper_helpers/ood_metrics/runner.py -- per the project's
testing policy (CLAUDE.md tier 4), paper_helpers scripts get smoke coverage only:
import + one call on tiny synthetic data, not full behavioral coverage.
"""
import numpy as np
import pandas as pd
import pytest

from src.paper_helpers.ood_metrics.runner import DATASET_CSV_FILENAMES, run_ood_comparison


def _write_fake_predictions_csv(path, n=20, seed=0):
    """Callback schema (src/callbacks/test_artifacts_callback.py): MSP under
    `prediction_prob_score`."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(alpha=[1, 1], size=n)
    df = pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)],
        "target": rng.integers(0, 2, size=n),
        "prediction": probs.argmax(axis=1),
        "prediction_prob_score": probs.max(axis=1),
        "class_probs": [list(p) for p in probs],
        "fold": ["test"] * n,
    })
    df.to_csv(path, index=False)


def _write_fake_inference_csv(path, n=20, seed=0):
    """Inference schema (src/inference/records.py): MSP under `confidence`, plus the
    `class_logits` that `score_mode="dempster_shafer"` needs. This is the schema every
    CIFAR-100 prediction CSV uses."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2.0, size=(n, 2))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    df = pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)],
        "target": rng.integers(0, 2, size=n),
        "prediction": probs.argmax(axis=1),
        "confidence": probs.max(axis=1),
        "class_logits": [str(row.tolist()) for row in logits],
        "class_probs": [str(row.tolist()) for row in probs],
        "fold": ["test"] * n,
    })
    df.to_csv(path, index=False)


def test_run_ood_comparison_smoke(tmp_path):
    method_dir = tmp_path / "baseline"
    method_dir.mkdir()
    _write_fake_predictions_csv(method_dir / DATASET_CSV_FILENAMES["acevedo"], seed=1)
    _write_fake_predictions_csv(method_dir / DATASET_CSV_FILENAMES["wong"], seed=2)

    out_dir = tmp_path / "out"
    df = run_ood_comparison(
        dataset="acevedo",
        methods={"Baseline": method_dir},
        ood_datasets=["wong"],
        out_dir=out_dir,
        score_mode="msp",
    )

    assert list(df["Method"]) == ["Baseline"]
    assert "wong" in df.columns
    assert (out_dir / "acevedo_results.csv").exists()
    # The default estimator stays the frozen subsample path, whose cells are
    # "mean ± std" strings -- published ISBI numbers depend on it.
    assert isinstance(df.iloc[0]["wong"], str)


def _inference_method_dir(tmp_path, name="baseline"):
    method_dir = tmp_path / name
    method_dir.mkdir()
    _write_fake_inference_csv(method_dir / DATASET_CSV_FILENAMES["acevedo"], seed=1)
    _write_fake_inference_csv(method_dir / DATASET_CSV_FILENAMES["wong"], seed=2)
    return method_dir


def test_full_population_returns_float_cells_for_the_inference_schema(tmp_path):
    """Covers both halves of the CIFAR-100 change at once: the `confidence` fallback
    (this schema has no `prediction_prob_score`, so it raised before) and the
    full-population estimator."""
    out_dir = tmp_path / "out"
    df = run_ood_comparison(
        dataset="acevedo",
        methods={"Baseline": _inference_method_dir(tmp_path)},
        ood_datasets=["wong"],
        out_dir=out_dir,
        score_mode="msp",
        estimator="full_population",
    )

    assert isinstance(df.iloc[0]["wong"], float)
    assert (out_dir / "acevedo_results.csv").exists()


def test_emits_one_file_per_score_mode(tmp_path):
    """Pins the "call the runner twice" decision (src/paper_helpers/ood_metrics/
    cifar100.py) against someone later re-deciding for a list-valued score_mode."""
    method_dir = _inference_method_dir(tmp_path)
    out_dir = tmp_path / "out"
    values = {}
    for score_mode in ("msp", "dempster_shafer"):
        df = run_ood_comparison(
            dataset="acevedo",
            methods={"Baseline": method_dir},
            ood_datasets=["wong"],
            out_dir=out_dir,
            score_mode=score_mode,
            out_filename=f"acevedo_{score_mode}.csv",
            estimator="full_population",
        )
        assert (out_dir / f"acevedo_{score_mode}.csv").exists()
        values[score_mode] = df.iloc[0]["wong"]

    assert values["msp"] != values["dempster_shafer"]


def test_unknown_estimator_raises(tmp_path):
    """A typo must not silently fall through to the frozen path."""
    with pytest.raises(ValueError, match="full_population"):
        run_ood_comparison(
            dataset="acevedo",
            methods={"Baseline": _inference_method_dir(tmp_path)},
            ood_datasets=["wong"],
            out_dir=tmp_path / "out",
            estimator="bootstrp",
        )
