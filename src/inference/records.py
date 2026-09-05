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
from datetime import datetime
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


def _reduce_uncertainty(uncertainty: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Collapse a per-class uncertainty tensor to one scalar per sample.

    SNGP and DeepEnsemble already emit `[B, 1]`, but `BaselineClassifier.mc_predict`
    returns the per-class std of the stacked per-pass logits, i.e. `[B, C]`. Reducing
    by the mean over classes matches both existing precedents --
    `src/inference/predict_image.py` (`std.mean(dim=1)`) and `DeepEnsemble.forward`
    (`.var(dim=0).mean(dim=-1, keepdim=True)`).
    """
    if uncertainty is None:
        return None
    if uncertainty.dim() == 2 and uncertainty.shape[1] > 1:
        return uncertainty.mean(dim=1, keepdim=True)
    return uncertainty


def extract_model_outputs(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    use_mc_dropout: bool,
    mc_passes: int,
    capture_members: bool = False,
) -> BatchOutputs:
    """Run one batch through `model` and collect everything a record may need.

    MC-Dropout takes precedence when the net supports it (`mc_predict` is only defined
    on `BaselineClassifier`); note it returns the *mean* of per-pass logits alongside
    the *mean of per-pass softmax*, so `softmax(logits) != probs` for MC runs. That
    asymmetry is inherent to the estimator, not a bug -- see `docs/KNOWN_ISSUES.md`.

    `capture_members=True` additionally populates `member_logits` -- `[M, B, C]`,
    ensemble members or MC-Dropout passes (the same shape convention either way, so
    `build_records` doesn't need to know which). For an ensemble this is free (already
    on `ModelOutput`); for MC-Dropout it means calling `mc_forward_samples` directly
    instead of `mc_predict`, which runs the same `T` forward passes either way -- never
    both -- and reduces them here rather than inside `mc_predict`.
    """
    if use_mc_dropout and hasattr(model, "mc_predict"):
        if capture_members and hasattr(model, "mc_forward_samples"):
            logits_stack = model.mc_forward_samples(x, T=mc_passes)  # [T, B, C]
            probs_stack = torch.softmax(logits_stack, dim=-1)
            uncertainty = logits_stack.std(dim=0, unbiased=False)
            return BatchOutputs(
                logits=logits_stack.mean(dim=0),
                probs=probs_stack.mean(dim=0),
                uncertainty=_reduce_uncertainty(uncertainty),
                member_logits=logits_stack,
            )
        result = model.mc_predict(x, T=mc_passes, return_std=True, apply_softmax=True)
        if isinstance(result, tuple) and len(result) == 3:
            logits, probs, uncertainty = result
            return BatchOutputs(logits=logits, probs=probs, uncertainty=_reduce_uncertainty(uncertainty))

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
        uncertainty=_reduce_uncertainty(uncertainty),
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
    save_member_logits: bool = False,
) -> List[Dict[str, Any]]:
    """One `predictions.csv` row per sample in the batch.

    `class_logits` is always written: softmax is shift-invariant, so logits cannot be
    recovered from `class_probs` afterwards, and logit-magnitude metrics
    (Dempster-Shafer, energy, temperature scaling) are unrecoverable without them.
    `raw_logits` (SNGP's pre-mean-field head output) and `uncertainty` follow the
    optional-column pattern -- present only when the net emits them.

    `member_logits` (`[M, C]` per row -- ensemble members or MC-Dropout passes, the
    same shape either way) is written only when `save_member_logits=True` *and*
    `outputs.member_logits` is present, since it is far larger than every other column
    combined (`infer.save.save_member_logits` defaults to off for exactly this reason).

    All list-valued columns are `json.dumps`-encoded, so readers can `json.loads` them
    without the `ast.literal_eval` fallback the training-time callback's repr-encoded
    lists need.

    Column order follows first-seen-key order in these dicts, since
    `pd.DataFrame(records)` preserves it.
    """
    confs = outputs.probs.max(dim=1).values

    targets_cpu = targets.detach().cpu().tolist()
    preds_cpu = preds.detach().cpu().tolist()
    confs_cpu = confs.detach().cpu().tolist()
    probs_cpu = outputs.probs.detach().cpu().tolist()
    logits_cpu = outputs.logits.detach().cpu().tolist()

    raw_logits_cpu = None
    if outputs.raw_logits is not None:
        raw_logits_cpu = outputs.raw_logits.detach().cpu().tolist()

    unc_cpu = None
    if outputs.uncertainty is not None:
        unc = to_cpu_tensor(outputs.uncertainty).reshape(-1)
        if unc.numel() != len(image_ids):
            raise ValueError(
                f"uncertainty has {unc.numel()} values for {len(image_ids)} samples; expected one "
                "per sample. A per-class uncertainty must be reduced before record building "
                "(see _reduce_uncertainty)."
            )
        unc_cpu = unc.tolist()

    member_logits_cpu = None
    if save_member_logits and outputs.member_logits is not None:
        # [M, B, C] -> [B, M, C], so member_logits_cpu[idx] is this sample's [M, C].
        member_logits_cpu = outputs.member_logits.detach().cpu().permute(1, 0, 2).tolist()

    records: List[Dict[str, Any]] = []
    for idx in range(len(image_ids)):
        record = {
            "image_id": str(image_ids[idx]),
            "fold": str(fold[idx]) if fold is not None else "unknown",
            "target": int(targets_cpu[idx]),
            "prediction": int(preds_cpu[idx]),
            "confidence": float(confs_cpu[idx]),
            "class_logits": json.dumps(logits_cpu[idx]),
            "class_probs": json.dumps(probs_cpu[idx]),
            "stream": stream_name,
        }
        if raw_logits_cpu is not None:
            record["raw_logits"] = json.dumps(raw_logits_cpu[idx])
        if unc_cpu is not None:
            record["uncertainty"] = float(unc_cpu[idx])
        if member_logits_cpu is not None:
            record["member_logits"] = json.dumps(member_logits_cpu[idx])
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
    provenance: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write `predictions.csv`, `metrics.json` and `run.json` under `output_root`.

    Called from `_finalize`, so nothing is written for a run that crashed partway.
    """
    records = list(records)

    if _flag(save_cfg, "save_csv", True):
        df = pd.DataFrame(records)
        csv_path = output_root / "predictions.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")

    if _flag(save_cfg, "save_metrics_json", True):
        metrics_path = output_root / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(dict(metrics), f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

    if provenance is not None and _flag(save_cfg, "save_run_json", True):
        run_json = dict(provenance)
        run_json["written_at"] = datetime.now().isoformat(timespec="seconds")
        run_json["n_rows"] = len(records)
        # Column list so a caller can answer "does this run have logits?" without
        # opening a multi-MB CSV.
        run_json["columns"] = list(records[0]) if records else []
        run_path = output_root / "run.json"
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(run_json, f, indent=2, default=str)
        logger.info(f"Saved run provenance to {run_path}")
