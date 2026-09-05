"""Metric registry: maps a plain string key to a function that turns a loaded
`PredictionFrame` (plus optional OOD frames and options) into result rows.

Mirrors `src/models/registry.py`'s shape -- a module-level dict plus a decorator that
raises on a name collision and lists known names on a miss -- so a new metric is
"write the function, decorate it", not "add a branch to some dispatcher". This module
only defines the registry itself; implementations live in `src/metrics/registered.py`
(see that module's docstring for why importing `src.metrics` is enough to populate
this dict regardless of import order).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional

from src.metrics.io import PredictionFrame


@dataclass(frozen=True)
class MetricContext:
    """Everything a metric function is handed.

    `run` carries whatever identifying info the caller has about this prediction file
    (e.g. `src/metrics/run_metrics.py` puts `label`/`run_dir`/`dataset` there), kept as
    a loosely-typed `Mapping` here rather than a concrete class so this module doesn't
    need to depend on any particular driver's notion of run identity.
    """

    frame: PredictionFrame
    ood_frames: Mapping[str, PredictionFrame] = field(default_factory=dict)
    run: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRow:
    """One output row.

    `value` is the machine-readable number a metric produced. `value_str` is
    additionally set only for a bootstrap summary that already has a canonical
    "mean ± std" rendering -- so published formatting (`csv/ood_metrics/*.csv`,
    `docs/RESULTS.md`) stays reproducible for those, while a plain point metric
    leaves it `None`. `scope` distinguishes multiple rows from one metric run on one
    frame -- an OOD dataset name for a cross-dataset metric, a stream name for a
    per-stream one, or `"run"` for a single-frame summary.
    """

    metric: str
    value: Optional[float]
    std: Optional[float] = None
    value_str: Optional[str] = None
    scope: str = "run"
    extra: Mapping[str, Any] = field(default_factory=dict)


MetricFn = Callable[[MetricContext], List[MetricRow]]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    fn: MetricFn
    requires: FrozenSet[str] = frozenset()
    needs_ood: bool = False
    description: str = ""


METRIC_REGISTRY: Dict[str, MetricSpec] = {}


def register_metric(
    name: str,
    *,
    requires: Any = (),
    needs_ood: bool = False,
    description: str = "",
) -> Callable[[MetricFn], MetricFn]:
    """Function decorator: registers `fn` under `name`. Returns `fn` unchanged, so a
    decorated metric function can still be called/tested directly.

    `requires` names the `PredictionFrame.capabilities` this metric needs (e.g.
    `{"logits"}`) -- a driver compares this against `frame.capabilities` and skips
    with a reason rather than calling `fn` and hitting a `MissingPredictionData`
    partway through. `needs_ood` marks a metric that only makes sense with at least
    one OOD frame (a run with no sibling dataset directories on disk has none).
    """

    def decorator(fn: MetricFn) -> MetricFn:
        if name in METRIC_REGISTRY and METRIC_REGISTRY[name].fn is not fn:
            raise ValueError(f"Metric name '{name}' is already registered to a different function.")
        METRIC_REGISTRY[name] = MetricSpec(
            name=name,
            fn=fn,
            requires=frozenset(requires),
            needs_ood=needs_ood,
            description=description,
        )
        return fn

    return decorator


def get_metric(name: str) -> MetricSpec:
    """Look up a registered metric by name."""
    if name not in METRIC_REGISTRY:
        raise KeyError(f"Unknown metric '{name}'. Registered: {sorted(METRIC_REGISTRY)}")
    return METRIC_REGISTRY[name]
