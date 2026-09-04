"""Shared inference write-path: model-output extraction, per-sample record building,
and result serialization.

Both inference runners -- `ClassificationInferenceRunner`
(`src/inference/infer.py`) and `ArtifactInferenceRunner`
(`src/inference/infer_artifact.py`) -- produce the same `predictions.csv` schema and
the same `metrics.json`, so the code that decides *what a row contains* lives here
once rather than in two copies that drift (the two modules also disagree on
indentation, which makes copy-paste between them a `TabError`).

This module is a leaf: it imports from neither runner, so it does not participate in
the existing `infer_artifact` -> `infer` import edge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import torch
from loguru import logger
from torchmetrics.classification import (
    Accuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.models.outputs import ModelOutput


@dataclass(frozen=True)
class BatchOutputs:
    """Everything one forward pass yields that a prediction record may draw on.

    `logits`/`probs` are always `[B, C]`. `uncertainty` is `[B, 1]` after the
    reduction in `extract_model_outputs`. `raw_logits` is SNGP's pre-mean-field head
    output. `member_logits` is `[M, B, C]` -- ensemble members or MC-Dropout passes,
    which are the same thing downstream (see `build_records`).
    """

    logits: torch.Tensor
    probs: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    raw_logits: Optional[torch.Tensor] = None
    member_logits: Optional[torch.Tensor] = None


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_name)


def to_cpu_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.as_tensor(x).detach().cpu()


def build_metrics(num_classes: int, metric_names: Sequence[str]) -> Dict[str, Any]:
    """Instantiate the torchmetrics objects named in `infer.metrics.items`.

    `nll` and `brier` are absent by design -- both are computed from the collected
    records at finalize time, not accumulated per batch.
    """
    metric_names = set(metric_names)
    metrics: Dict[str, Any] = {}

    if "acc" in metric_names:
        metrics["acc"] = Accuracy(task="multiclass", num_classes=num_classes)
    if "ece" in metric_names:
        metrics["ece"] = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm="l1")
    if "precision" in metric_names:
        metrics["precision"] = MulticlassPrecision(num_classes=num_classes, average="macro")
    if "recall" in metric_names:
        metrics["recall"] = MulticlassRecall(num_classes=num_classes, average="macro")
    if "f1" in metric_names:
        metrics["f1"] = MulticlassF1Score(num_classes=num_classes, average="macro")
    if "precision_micro" in metric_names:
        metrics["precision_micro"] = MulticlassPrecision(num_classes=num_classes, average="micro")
    if "recall_micro" in metric_names:
        metrics["recall_micro"] = MulticlassRecall(num_classes=num_classes, average="micro")
    if "f1_micro" in metric_names:
        metrics["f1_micro"] = MulticlassF1Score(num_classes=num_classes, average="micro")
    if "auroc" in metric_names:
        metrics["auroc"] = MulticlassAUROC(num_classes=num_classes, average="macro")
    if "auprc" in metric_names:
        metrics["auprc"] = MulticlassAveragePrecision(num_classes=num_classes, average="macro")

    return metrics


def extract_model_outputs(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    use_mc_dropout: bool,
    mc_passes: int,
) -> BatchOutputs:
    """Run one batch through `model` and collect everything a record may need.

    MC-Dropout takes precedence when the net supports it (`mc_predict` is only defined
    on `BaselineClassifier`); note it returns the *mean* of per-pass logits alongside
    the *mean of per-pass softmax*, so `softmax(logits) != probs` for MC runs. That
    asymmetry is inherent to the estimator, not a bug -- see `docs/KNOWN_ISSUES.md`.
    """
    if use_mc_dropout and hasattr(model, "mc_predict"):
        result = model.mc_predict(x, T=mc_passes, return_std=True, apply_softmax=True)
        if isinstance(result, tuple) and len(result) == 3:
            logits, probs, uncertainty = result
            return BatchOutputs(logits=logits, probs=probs, uncertainty=uncertainty)

    output = model(x)

    uncertainty = None
    raw_logits = None
    member_logits = None
    if isinstance(output, ModelOutput):
        logits = output.logits
        uncertainty = output.variance
        raw_logits = output.raw_logits
        member_logits = output.member_logits
    elif isinstance(output, tuple):
        # Legacy nets returning a bare (mean_field_logits, raw_logits, pred_var) tuple.
        logits = output[0]
        if len(output) >= 2 and torch.is_tensor(output[1]):
            raw_logits = output[1]
        if len(output) >= 3 and torch.is_tensor(output[2]):
            uncertainty = output[2]
    else:
        logits = output

    probs = torch.softmax(logits, dim=1)
    return BatchOutputs(
        logits=logits,
        probs=probs,
        uncertainty=uncertainty,
        raw_logits=raw_logits,
        member_logits=member_logits,
    )


def build_records(
    *,
    image_ids: Sequence[Any],
    fold: Optional[Sequence[Any]],
    targets: torch.Tensor,
    preds: torch.Tensor,
    outputs: BatchOutputs,
    stream_name: str,
) -> List[Dict[str, Any]]:
    """One `predictions.csv` row per sample in the batch.

    Column order follows first-seen-key order in these dicts, since
    `pd.DataFrame(records)` preserves it.
    """
    confs = outputs.probs.max(dim=1).values

    targets_cpu = targets.detach().cpu().tolist()
    preds_cpu = preds.detach().cpu().tolist()
    confs_cpu = confs.detach().cpu().tolist()
    probs_cpu = outputs.probs.detach().cpu().tolist()

    unc_cpu = None
    if outputs.uncertainty is not None:
        unc_cpu = to_cpu_tensor(outputs.uncertainty).reshape(-1).tolist()

    records: List[Dict[str, Any]] = []
    for idx in range(len(image_ids)):
        record = {
            "image_id": str(image_ids[idx]),
            "fold": str(fold[idx]) if fold is not None else "unknown",
            "target": int(targets_cpu[idx]),
            "prediction": int(preds_cpu[idx]),
            "confidence": float(confs_cpu[idx]),
            "class_probs": json.dumps(probs_cpu[idx]),
            "stream": stream_name,
        }
        if unc_cpu is not None:
            record["uncertainty"] = float(unc_cpu[idx])
        records.append(record)
    return records


def _flag(save_cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read a boolean from a plain dict or a `DictConfig` `infer.save` section."""
    try:
        value = save_cfg[key]
    except (KeyError, AttributeError):
        return default
    return default if value is None else bool(value)


def write_outputs(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    save_cfg: Mapping[str, Any],
) -> None:
    """Write `predictions.csv` and `metrics.json` under `output_root`, per `save_cfg`."""
    if _flag(save_cfg, "save_csv", True):
        df = pd.DataFrame(list(records))
        csv_path = output_root / "predictions.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")

    if _flag(save_cfg, "save_metrics_json", True):
        metrics_path = output_root / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(dict(metrics), f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
