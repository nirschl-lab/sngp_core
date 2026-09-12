"""Shared prediction-CSV fixture writer for tests/metrics/* and tests/paper_helpers/*.

Both schemas this project uses are covered -- "inference"
(`src/inference/records.py`: `confidence`/`class_probs`/`stream`[,`class_logits`,
`raw_logits`,`uncertainty`,`member_logits`], JSON-encoded lists) and "callback"
(`src/callbacks/test_artifacts_callback.py`: `prediction_prob_score`/
`true_bin_label`/`class_logits`/`class_probs`, repr-encoded lists) -- see
`.claude/skills/metrics/references/csv_schema.md`.

The optional `with_*` flags let one fixture stand in for any schema version: leaving
them all off reproduces a pre-schema-2 run (the shape every CSV on disk had before
2026-09-04), which is what the `requires=`/`status=skipped` paths need to exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def write_predictions_csv(
    path: Path,
    *,
    n: int = 4,
    num_classes: int = 2,
    schema: str = "inference",
    seed: int = 0,
    confidences: Optional[Sequence[float]] = None,
    all_correct: bool = False,
    fold: str = "test",
    stream: str = "default",
    with_logits: bool = False,
    with_raw_logits: bool = False,
    with_member_logits: bool = False,
    with_uncertainty: bool = False,
    uncertainty_kind: str = "gp_predictive_variance",
    with_comparable_uncertainty: bool = False,
    with_decomposition: bool = False,
    members: int = 3,
) -> pd.DataFrame:
    """Write a synthetic predictions.csv at `path` and return the DataFrame written.

    If `confidences` is given (length `n`), each row's max-class probability is
    pinned to that value, with the rest of the probability mass split evenly across
    the other classes -- useful for OOD-AUROC fixtures that need a controlled
    confidence level (e.g. an obviously-confident ID set vs an obviously-uncertain
    OOD one). Otherwise probabilities are drawn from a `Dirichlet(1,...,1)`, seeded
    by `seed`.

    `all_correct=True` sets `target` equal to the argmax prediction for every row
    (deterministic 100% accuracy) instead of an independent random draw -- useful for
    a test that asserts on a known accuracy value.
    """
    rng = np.random.default_rng(seed)

    if confidences is not None:
        if len(confidences) != n:
            raise ValueError(f"confidences must have length n={n}, got {len(confidences)}")
        probs = np.zeros((n, num_classes))
        for i, c in enumerate(confidences):
            rest = (1.0 - c) / (num_classes - 1) if num_classes > 1 else 0.0
            probs[i] = rest
            probs[i, i % num_classes] = c
    else:
        probs = rng.dirichlet(alpha=[1] * num_classes, size=n)

    preds = probs.argmax(axis=1)
    targets = preds.copy() if all_correct else rng.integers(0, num_classes, size=n)

    if schema == "inference":
        data = {
            "image_id": [f"img{i}" for i in range(n)],
            "fold": [fold] * n,
            "target": targets,
            "prediction": preds,
            "confidence": probs.max(axis=1),
            "class_probs": [json.dumps(p.tolist()) for p in probs],
            "stream": [stream] * n,
        }
        if with_logits:
            logits = rng.normal(size=(n, num_classes))
            data["class_logits"] = [json.dumps(row.tolist()) for row in logits]
        if with_raw_logits:
            raw_logits = rng.normal(size=(n, num_classes))
            data["raw_logits"] = [json.dumps(row.tolist()) for row in raw_logits]
        if with_member_logits:
            member_logits = rng.normal(size=(n, members, num_classes))
            data["member_logits"] = [json.dumps(row.tolist()) for row in member_logits]
        if with_uncertainty:
            data["uncertainty"] = rng.gamma(shape=2.0, scale=0.5, size=n)
            data["uncertainty_kind"] = [uncertainty_kind] * n
        if with_comparable_uncertainty:
            # Real values for the always-written schema-3 columns, so a metric reading
            # them gets something consistent with `class_probs` rather than noise.
            clipped = np.clip(probs, 1e-12, 1.0)
            data["predictive_entropy"] = -(clipped * np.log(clipped)).sum(axis=1)
            ordered = np.sort(probs, axis=1)
            data["confidence_margin"] = ordered[:, -1] - (ordered[:, -2] if num_classes > 1 else 0.0)
            ds_logits = np.log(clipped)
            data["dempster_shafer"] = num_classes / (np.exp(ds_logits).sum(axis=1) + num_classes)
        if with_decomposition:
            aleatoric = rng.uniform(0.0, 0.5, size=n)
            epistemic = rng.uniform(0.0, 0.2, size=n)
            data["total_entropy"] = aleatoric + epistemic
            data["aleatoric_entropy"] = aleatoric
            data["mutual_information"] = epistemic
    elif schema == "callback":
        logits = rng.normal(size=(n, num_classes))
        data = {
            "image_id": [f"img{i}" for i in range(n)],
            "target": targets,
            "prediction": preds,
            "prediction_prob_score": probs.max(axis=1),
            "true_bin_label": (preds == targets).astype(int),
            # Real list objects, not JSON strings -- pandas repr-stringifies these on
            # to_csv, matching TestArtifactsCallback's own DataFrame construction.
            "class_logits": [row.tolist() for row in logits],
            "class_probs": [row.tolist() for row in probs],
            "fold": [fold] * n,
        }
    else:
        raise ValueError(f"Unsupported schema: {schema!r}")

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df
