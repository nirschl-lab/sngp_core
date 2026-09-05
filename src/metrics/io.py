"""Shared loader for this project's two prediction-CSV schemas.

Two pipelines write prediction CSVs with different column names for the same
underlying quantities -- see `.claude/skills/metrics/references/csv_schema.md`:

- **"callback"** schema: `src/callbacks/test_artifacts_callback.py`, columns
  `prediction_prob_score`/`true_bin_label`/`class_logits` (repr-encoded lists).
- **"inference"** schema: `src/inference/records.py`, columns
  `confidence`/`stream`/`class_logits` (JSON-encoded lists).

`load_predictions()` detects which schema a file is, normalizes both onto the same
derived columns, and reports which arrays are actually available (`capabilities`) so
a caller -- in particular the metric registry in `src/metrics/registry.py` -- can
decide whether to run or skip a metric *before* hitting a `KeyError` partway through.

This module intentionally has no dependency on anything else in `src.metrics` or
`src.paper_helpers` (`auc.py` and `artifact_quantification.py` import *from* here,
not the other way around), so importing it can never trigger a cycle.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


class MissingPredictionData(KeyError):
    """Raised when a requested column/array isn't in a prediction CSV.

    Subclasses `KeyError` so existing `except KeyError:` call sites keep working
    unchanged, while carrying a message that explains *why* the data is missing
    (wrong schema, or a run written before a column existed) rather than just which
    key lookup failed.
    """


REAL_STREAM_CANDIDATES = {"real", "clean", "original", "id", "in", "in_distribution"}
ARTIFACT_STREAM_CANDIDATES = {"artifact", "artifacts", "simulated", "ood", "out", "out_distribution"}


def canonicalize_stream(value: Any) -> str:
    """Map a `stream` value to `"real"`/`"artifact"`, or lowercase it unchanged if it
    matches neither set. Exact copy of
    `src.paper_helpers.ood_metrics.artifact_quantification._canonicalize_stream` --
    that module keeps its own private copy for now (see the module docstring there),
    so this is the second, not the only, definition."""
    x = str(value).strip().lower()
    if x in REAL_STREAM_CANDIDATES:
        return "real"
    if x in ARTIFACT_STREAM_CANDIDATES:
        return "artifact"
    return x


def parse_float_list(value: Any) -> Optional[List[float]]:
    """Parse a `class_probs`/`class_logits` cell into a list of floats.

    Tries JSON first (the inference schema's convention), then falls back to
    `ast.literal_eval` (the training-time callback schema's repr-encoded lists).
    Returns `None` for anything that isn't a list or can't be parsed as one -- except
    a malformed Python-literal string, which raises, matching this project's
    long-standing parser behavior (`src.metrics.auc._parse_class_probs` /
    `src.paper_helpers.ood_metrics.artifact_quantification._parse_class_probs`, now
    both thin aliases for this function) so existing callers see no behavior change.
    """
    if isinstance(value, list):
        return [float(v) for v in value]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [float(v) for v in parsed]

    return None


def parse_nested_float_list(value: Any) -> Optional[List[List[float]]]:
    """Parse a `member_logits` cell (`[M, C]`, JSON-encoded) into nested floats.

    No legacy precedent to match -- `member_logits` is new
    (`infer.save.save_member_logits`) -- so unlike `parse_float_list` this simply
    returns `None` on any parse failure rather than propagating one.
    """
    if isinstance(value, list):
        return [[float(v) for v in row] for row in value]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return None
        if isinstance(parsed, list):
            try:
                return [[float(v) for v in row] for row in parsed]
            except TypeError:
                return None

    return None


def normalized_entropy(probs: Sequence[float], eps: float = 1e-12) -> float:
    """Entropy normalized to `[0, 1]` by `log(num_classes)` -- an OOD score where
    larger means more uncertain. Exact copy of `src.metrics.auc._normalized_entropy`."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)
    k = p.size
    if k <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(k))


def shannon_entropy_nats(probs: Sequence[float], eps: float = 1e-12) -> float:
    """Shannon entropy in nats, unnormalized -- deliberately distinct from
    `normalized_entropy` (see `src.visualization.predictive_entropy._raw_entropy_from_probs`,
    now a thin alias for this function): dividing by `log(num_classes)` makes cross-dataset
    OOD scores comparable, but throws away the raw scale a KDE plot wants to show."""
    p = np.asarray(probs, dtype=float)
    return float(-np.sum(p * np.log(p + eps)))


def filter_fold(df: pd.DataFrame, fold: Optional[str], source: Union[str, Path]) -> pd.DataFrame:
    """Filter to one `fold` value, or return `df` unchanged if `fold` is `None`.

    Matches the exact `KeyError`/`ValueError` shape both
    `src.metrics.calculate_ood_metrics` and `src.visualization.predictive_entropy`
    already raised as copy-pasted inline blocks -- now shared instead of duplicated.
    """
    if fold is None:
        return df
    if "fold" not in df.columns:
        raise KeyError(f"{source} has no 'fold' column to filter on.")
    filtered = df[df["fold"] == fold]
    if filtered.empty:
        raise ValueError(f"No rows with fold={fold!r} in {source}")
    return filtered


@dataclass(frozen=True)
class OodFoldPolicy:
    """Which fold to filter the in-distribution vs out-of-distribution frame to,
    before sampling for OOD AUROC. `None` means "don't filter"."""

    id_fold: Optional[str]
    ood_fold: Optional[str]


# `src.metrics.auc.AUROC_across_dataset` has always filtered the ID frame to
# fold=='test' and left the OOD frame unfiltered -- an asymmetry that is load-bearing
# for every published number in csv/final/ and csv/isbi_test_files/. This constant
# reproduces that behavior exactly; it is the default so nothing published moves.
LEGACY_ISBI_FOLD_POLICY = OodFoldPolicy(id_fold="test", ood_fold=None)

# Filters both frames the same way -- the fold policy an analysis should reach for
# deliberately, not by default.
SYMMETRIC_FOLD_POLICY = OodFoldPolicy(id_fold="test", ood_fold="test")


@dataclass(frozen=True)
class PredictionFrame:
    """A loaded, schema-normalized prediction CSV.

    `df` carries the original columns plus the derived ones documented on
    `load_predictions`. `capabilities` is the set of things a metric can rely on
    being present -- check it (or pass `require=` to `load_predictions`) before
    reaching for an array that a given run may not have.
    """

    df: pd.DataFrame
    schema: Literal["callback", "inference"]
    path: Path
    num_classes: int
    capabilities: FrozenSet[str]
    streams: Tuple[str, ...]


def _stack_column(frame: PredictionFrame, column: str, source_name: str) -> np.ndarray:
    values = frame.df[column].tolist()
    n_bad = sum(1 for v in values if v is None)
    if n_bad:
        raise ValueError(f"{frame.path}: {n_bad} row(s) failed to parse '{source_name}' into a list of floats.")
    return np.asarray(values, dtype=float)


def probs_array(frame: PredictionFrame) -> np.ndarray:
    """`[N, C]` softmax probabilities. Always available -- both schemas require it."""
    return _stack_column(frame, "probs", "class_probs")


def logits_array(frame: PredictionFrame) -> np.ndarray:
    """`[N, C]` raw logits.

    Softmax is shift-invariant, so logits cannot be recovered from `class_probs`
    after the fact -- a run written before `predictions_csv_schema: 2`
    (`src/inference/records.py`) has no logits, and re-running inference is the only
    fix, not a converter.
    """
    if "logits" not in frame.capabilities:
        raise MissingPredictionData(
            f"{frame.path} has no 'class_logits' column. Runs written before "
            "predictions_csv_schema=2 did not persist logits; softmax probabilities "
            "cannot be inverted (softmax is shift-invariant). Re-run inference for "
            "this checkpoint/dataset."
        )
    return _stack_column(frame, "logits", "class_logits")


def member_logits_array(frame: PredictionFrame) -> np.ndarray:
    """`[N, M, C]` per-member/per-pass logits (ensemble members or MC-Dropout passes --
    same shape convention either way). Only present when the run was written with
    `infer.save.save_member_logits=true`."""
    if "member_logits" not in frame.capabilities:
        raise MissingPredictionData(
            f"{frame.path} has no 'member_logits' column. Re-run inference with "
            "infer.save.save_member_logits=true to persist per-member/per-pass logits."
        )
    return _stack_column(frame, "member_logits_parsed", "member_logits")


def load_predictions(
    path: Union[str, Path],
    *,
    fold: Optional[str] = None,
    require: Sequence[str] = (),
) -> PredictionFrame:
    """Load one prediction CSV, detect its schema, and normalize it onto shared columns.

    Derived columns, additive -- the raw `class_probs`/`class_logits`/`member_logits`
    string columns are left untouched alongside them:

    - `confidence`: max softmax probability, whichever of `confidence` /
      `prediction_prob_score` the file has.
    - `correct`: from `true_bin_label` if present, else `prediction == target`.
    - `probs`: `class_probs` parsed into a Python list of floats per row.
    - `logits`: `class_logits` parsed the same way, only when that column exists.
    - `member_logits_parsed`: `member_logits` parsed into `[M, C]` nested floats, only
      when that column exists (kept under a different name than the raw `member_logits`
      string column it's parsed from, since inference already uses that exact name).
    - `entropy_norm` / `entropy_nats`: `normalized_entropy`/`shannon_entropy_nats` of
      `probs`.
    - `stream_canonical`: `canonicalize_stream` of `stream`, defaulting to `"default"`
      when the file has no `stream` column at all.

    `capabilities` is a subset of `{"probs", "targets", "logits", "member_logits",
    "raw_logits", "uncertainty", "paired_streams"}` -- `"probs"`/`"targets"` are always
    present (both schemas require them); the rest depend on which optional columns the
    run happens to have. Never keyed on the presence of `stream` alone, since the
    inference schema always writes it (`"default"` for a non-artifact run) -- see
    `.claude/skills/metrics/references/csv_schema.md`.

    Raises `MissingPredictionData` (a `KeyError` subclass) if the file has neither
    schema's confidence column, or if `require` names a capability this file lacks.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No predictions file at {path}")

    raw = pd.read_csv(path)

    has_confidence = "confidence" in raw.columns
    has_prob_score = "prediction_prob_score" in raw.columns
    if has_confidence and has_prob_score:
        raise ValueError(
            f"{path} has both 'confidence' and 'prediction_prob_score' columns -- these "
            "are the inference-schema and callback-schema names for the same quantity, "
            "and a real prediction CSV should only ever have one. Check which pipeline "
            "actually wrote this file."
        )
    if has_confidence:
        schema: Literal["callback", "inference"] = "inference"
    elif has_prob_score:
        schema = "callback"
    else:
        raise MissingPredictionData(
            f"{path} has neither 'confidence' (inference schema) nor 'prediction_prob_score' "
            "(callback schema) -- can't tell which prediction-CSV schema this is. See "
            ".claude/skills/metrics/references/csv_schema.md."
        )

    if "class_probs" not in raw.columns:
        raise MissingPredictionData(f"{path} has no 'class_probs' column.")

    df = filter_fold(raw, fold, source=path).copy()

    if schema == "inference":
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    else:
        df["confidence"] = pd.to_numeric(df["prediction_prob_score"], errors="coerce")

    if "true_bin_label" in df.columns:
        df["correct"] = pd.to_numeric(df["true_bin_label"], errors="coerce") == 1
    else:
        df["correct"] = pd.to_numeric(df["prediction"], errors="coerce") == pd.to_numeric(
            df["target"], errors="coerce"
        )

    df["probs"] = df["class_probs"].map(parse_float_list)
    num_classes_values = df["probs"].map(lambda p: len(p) if p is not None else None).dropna()
    if num_classes_values.empty:
        raise ValueError(f"Could not determine num_classes from 'class_probs' in {path} -- no row parsed.")
    num_classes = int(num_classes_values.iloc[0])

    df["entropy_norm"] = df["probs"].map(lambda p: normalized_entropy(p) if p is not None else None)
    df["entropy_nats"] = df["probs"].map(lambda p: shannon_entropy_nats(p) if p is not None else None)

    capabilities = {"probs", "targets"}

    if "class_logits" in df.columns:
        df["logits"] = df["class_logits"].map(parse_float_list)
        capabilities.add("logits")

    if "raw_logits" in df.columns:
        capabilities.add("raw_logits")

    if "member_logits" in df.columns:
        df["member_logits_parsed"] = df["member_logits"].map(parse_nested_float_list)
        capabilities.add("member_logits")

    if "uncertainty" in df.columns:
        capabilities.add("uncertainty")

    if "stream" in df.columns:
        df["stream_canonical"] = df["stream"].map(canonicalize_stream)
    else:
        df["stream_canonical"] = "default"

    streams = tuple(sorted(df["stream_canonical"].unique().tolist()))
    if "real" in streams and "artifact" in streams:
        capabilities.add("paired_streams")

    frame = PredictionFrame(
        df=df,
        schema=schema,
        path=path,
        num_classes=num_classes,
        capabilities=frozenset(capabilities),
        streams=streams,
    )

    missing = set(require) - frame.capabilities
    if missing:
        raise MissingPredictionData(
            f"{path} is missing required capabilities {sorted(missing)} (has {sorted(frame.capabilities)})."
        )

    return frame
