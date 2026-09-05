"""Shared prediction-CSV fixture writer for tests/metrics/* and tests/paper_helpers/*.

Both schemas this project uses are covered -- "inference"
(`src/inference/records.py`: `confidence`/`class_probs`/`stream`[,`class_logits`,
`raw_logits`,`uncertainty`,`member_logits`], JSON-encoded lists) and "callback"
(`src/callbacks/test_artifacts_callback.py`: `prediction_prob_score`/
`true_bin_label`/`class_logits`/`class_probs`, repr-encoded lists) -- see
`.claude/skills/metrics/references/csv_schema.md`.
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
