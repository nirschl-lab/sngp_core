# Testing: what to run for a given change

Four tiers of tests exist in this project (unit, config smoke, integration/slow,
untested-by-policy) — see
[docs/DEVELOPMENT.md#testing](../../docs/DEVELOPMENT.md#testing) for what each tier
covers and known flaky tests. This file is the operational rule for *which command to
run*, not what the tiers mean.

`make test`/`make test-full` run every test regardless of what changed — slow for
iterating on one module. Default to the narrowest command that covers the changed
paths; only fall back to the full suite for the "core" files below or right before
considering a task done.

**Path → test mapping**:

| Changed path | Run |
|---|---|
| `src/metrics/**` | `pytest tests/metrics/` |
| `src/models/backbones.py`, `src/models/components/spectral_norm.py` | `pytest tests/models/test_backbone_factory.py tests/models/baseline/test_backbones.py tests/models/sngp/test_spectral_norm.py` |
| `src/models/baseline/**`, `src/models/baseline_lit_module.py` | `pytest tests/models/baseline/ tests/models/test_output_contract.py` |
| `src/models/sngp/**`, `src/models/sngp_lit_module.py` | `pytest tests/models/sngp/` |
| `src/models/ensemble/**`, `src/models/deep_ensemble_lit_module.py` | `pytest tests/checkpointing/test_ensemble_assembly.py tests/models/test_output_contract.py` |
| `src/checkpointing/legacy.py`, `src/checkpointing/resolve.py` | `pytest tests/checkpointing/` |
| `src/data/classification_image_datamodule.py`, `artifact_image_datamodule.py`, `mnist_datamodule.py` | `pytest tests/test_datamodules.py` |
| `src/callbacks/**` | `pytest tests/callbacks/` |
| `src/inference/**` | `pytest tests/test_infer.py` |
| `src/train.py` | `pytest tests/test_train.py` |
| `src/eval.py` | `pytest tests/test_eval.py` |
| `src/paper_helpers/**` | `pytest tests/paper_helpers/` |
| `src/visualization/**` | untested by policy — no dedicated suite; smoke-test manually |
| `scripts/hf/export_to_hub.py` | `pytest tests/hf/` |
| `configs/data/*.yaml` | `pytest tests/test_configs.py::TestDatasetConfigDrift tests/test_datamodules.py` |
| `configs/model/*.yaml` | `pytest tests/test_configs.py::TestModelConfigs` |
| `configs/experiment/*.yaml` | `pytest tests/test_configs.py::TestExperimentClassFreqConsistency` |
| `configs/hparams_search/**` | `pytest tests/test_sweeps.py` |
| any other `configs/**` | `pytest tests/test_configs.py` |

**Core files — no safe scoped subset, run `make test` instead:** these are the
single-source-of-truth contracts from the architecture map
([docs/DEVELOPMENT.md#architecture-overview](../../docs/DEVELOPMENT.md#architecture-overview)),
load-bearing across every model family and dataset, so a scoped subset gives false
confidence — `src/models/lit_module_base.py`, `src/models/outputs.py`,
`src/models/registry.py`, `src/checkpointing/spec.py`, `src/checkpointing/io.py`,
`src/data/base_image_datamodule.py`, `src/utils/**`, `tests/conftest.py`,
`configs/paths/**`, `configs/trainer/**`.

Rules:
1. Map each changed file to its row (or to "core"); if a change spans multiple rows,
   union their commands rather than escalating to the full suite.
2. A path not in this table (new module/top-level dir) defaults to `make test` —
   don't guess a narrower scope the table doesn't cover.
3. Scoped runs keep the same fast/slow split as `make test`/`make test-full`
   (`pytest <paths> -k "not slow"` by default); include the slow tests in that same
   scope only when the change specifically touches integration/round-trip behavior
   there (e.g. checkpoint save/load logic → also run
   `tests/checkpointing/test_roundtrip.py`'s slow cases).
4. Scoped runs are for fast in-loop feedback, not a substitute for the real gate:
   still run `make test` (fast) before considering a change ready, and `make
   test-full` before considering the work genuinely done.
