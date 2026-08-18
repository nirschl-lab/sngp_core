"""Smoke test for src/paper_helpers/ood_metrics/runner.py -- per the project's
testing policy (CLAUDE.md tier 4), paper_helpers scripts get smoke coverage only:
import + one call on tiny synthetic data, not full behavioral coverage.
"""
import numpy as np
import pandas as pd

from src.paper_helpers.ood_metrics.runner import DATASET_CSV_FILENAMES, run_ood_comparison


def _write_fake_predictions_csv(path, n=20, seed=0):
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
