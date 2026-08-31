"""Smoke test for src/metrics/calculate_ood_metrics.py -- per the project's testing
policy, src/metrics/** gets covered by tests/metrics/.
"""
import json

import numpy as np
import pandas as pd

from src.metrics.calculate_ood_metrics import compute_ood_auroc


def _write_fake_predictions_csv(path, n=50, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(alpha=[1, 1], size=n)
    df = pd.DataFrame({
        "image_id": [f"img{i}" for i in range(n)],
        "fold": ["test"] * n,
        "target": rng.integers(0, 2, size=n),
        "prediction": probs.argmax(axis=1),
        "confidence": probs.max(axis=1),
        # JSON-encoded, matching the real infer.py `class_probs` column format.
        "class_probs": [json.dumps([float(v) for v in p]) for p in probs],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_compute_ood_auroc_smoke(tmp_path):
    run_dir = tmp_path / "run"
    _write_fake_predictions_csv(run_dir / "acevedo" / "predictions.csv", seed=1)
    _write_fake_predictions_csv(run_dir / "wong" / "predictions.csv", seed=2)

    df = compute_ood_auroc(run_dir, indist="acevedo", ood_datasets=["wong"], sample_rate=10)

    assert list(df["dataset"]) == ["wong"]
    assert {"msp_auroc", "entropy_auroc"}.issubset(df.columns)
    assert "±" in df.loc[0, "msp_auroc"]
    assert "±" in df.loc[0, "entropy_auroc"]
