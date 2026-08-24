# Hyperparameter Search Guide

How to tune each model family fairly before comparing Baseline, SNGP, and Deep
Ensemble against each other. The protocol here exists because every
`configs/experiment/*.yaml` previously hardcoded the same `lr`/`weight_decay`/
`batch_size` across all three families -- any observed difference between methods was
confounded with "this arbitrary config happened to suit this method better."

## Guiding principle

**Calibration and uncertainty are the evaluation axis for this project, never the
training-time selection axis.** Every model here is trained as a general classifier
and *then* evaluated for calibration/uncertainty quality -- tuning hyperparameters
against ECE, NLL, or predictive variance would bake calibration behavior into the
training methodology itself and defeat the point of measuring it independently. This
is why selection uses macro-AUPRC (a purely discriminative metric) while NLL/ECE are
logged every epoch as **read-only diagnostics** that never feed early stopping,
checkpointing, or the Optuna objective.

The same principle decides which SNGP knobs are tunable: `length_scale`,
`ridge_penalty`, and `rff_dim` are in the search space because with
`mean_field: True` (the project default) the training loss is computed on the
mean-field-corrected logits (`logits / sqrt(1 + predictive_variance)`, see
`src/models/sngp/sngp_classifier.py`), so these genuinely shape the *fit*. `mean_field`
and `cov_momentum` stay fixed -- those are pure uncertainty-mechanism knobs.

Dropout (`model.net.dropout_p`) is fixed at `0.2` for Baseline/Deep-Ensemble members,
never tuned: MC-Dropout at inference depends on it being nonzero, and tuning it
against a *deterministic* validation forward pass would push it toward 0, silently
breaking MC-Dropout without any test ever catching it. SNGP has no dropout parameter
at all, so it was never a shared knob to begin with.

## Prerequisites

- `.env` must set `EXPERIMENTS_HOME` and `PROJECT_NAME` (see `env_example`) --
  `configs/paths/default.yaml` resolves `log_dir` (and everything under it: Hydra
  multirun output, checkpoints, the Optuna sqlite study) as
  `${EXPERIMENTS_HOME}/${PROJECT_NAME}`. Hydra creates this directory itself if it
  doesn't exist yet -- **except** the `optuna/` subdirectory the sqlite storage lives
  in, which sqlite does not create on its own (`scripts/hpo/sweep.sh` handles this with
  `mkdir -p`; a bare local `-m hparams_search=...` invocation must do the same first).
- `sqlalchemy<2.0` is pinned in `pyproject.toml`: `optuna==2.10.1` (pinned transitively
  by `hydra-optuna-sweeper==1.2.0`) asserts on sqlalchemy 1.x engine internals and
  crashes with `AssertionError` in `get_current_version()` against sqlalchemy 2.x. Run
  `uv sync` after pulling this change.

## Selection metric

`val/auprc_best` -- the running max, across epochs, of macro-averaged
`MulticlassAveragePrecision` on the validation set (`LitModuleBase.val_auprc`/
`val_auprc_best`, computed in `on_validation_epoch_end`). Threshold-free and
imbalance-robust, which matters here: Tang is ~25:1 class imbalance
(`class_freq: [1909, 2428, 47702, 9331]`).

This exists specifically because `train()`'s `metric_dict` (`src/train.py`) is a
snapshot of `trainer.callback_metrics` -- the *last* epoch, not the best one. With
`EarlyStopping(patience=8)`, that's up to 8 epochs past the peak. Before this fix, an
Optuna objective reading any plain epoch-level metric would have ranked trials on
essentially random post-peak noise. `val/auprc_best` fixes this by tracking its own
max internally (a `torchmetrics.MaxMetric`), independent of when the run happens to
stop.

`val/nll` (plain, unweighted cross-entropy -- *not* the same as `val/loss`, which is
the class-balanced focal loss) and `val/ece` are logged alongside every epoch as
diagnostics. Check these after a sweep completes: if the AUPRC-optimal config comes
with a materially worse ECE/NLL than the untuned default, that's a real finding worth
reporting, not a bug to fix by adding calibration to the objective.

## Search space

Shared by all families (log-uniform where scale-free -- `tag(log, interval(...))` in
Optuna's override grammar):

| Param | Range | Current default |
|---|---|---|
| `model.optimizer.lr` | `tag(log, interval(1e-5, 1e-2))` | 1e-3 |
| `model.optimizer.weight_decay` | `tag(log, interval(1e-6, 1e-2))` | 1e-3 |
| `data.datamodule.batch_size` | `choice(64, 128, 256)` | 128 |

SNGP additionally (`configs/hparams_search/sngp.yaml` only):

| Param | Range | Current default |
|---|---|---|
| `model.net.length_scale` | `tag(log, interval(0.25, 16.0))` | 1.0 |
| `model.net.ridge_penalty` | `tag(log, interval(1e-4, 1e-1))` | 1e-3 |
| `model.net.rff_dim` | `choice(512, 1024, 2048)` | 1024 |

`rff_dim` costs O(d³) per forward -- a Cholesky solve on a `rff_dim x rff_dim` matrix
runs on every batch (`RandomFeatureGaussianProcess.forward`). 2048 is ~8x the head
cost of 1024, so equal *trial* budget across this space is not equal *compute*
budget; drop 2048 from the `choice(...)` list if wall-clock becomes the binding
constraint.

Deep Ensemble is **not swept independently** -- it inherits the tuned Baseline config,
since ensemble members are baseline models differing only in init seed
(Lakshminarayanan et al.). Sweeping it separately would cost N× a baseline trial for
a search that would likely land near the baseline optimum anyway.

## Fixed protocol constants

Identical across every sweep, never in the search space: `net.arch: resnet18`,
`pretrained: false`, `img_augmentations: light_augmentations`, `cb_beta: 0.999`,
`focal_gamma: 2.0`, `class_freq` (from the dataset's own config), the
`CosineAnnealingLR` family (`CosineAnnealingWarmRestarts` for Deep Ensemble, which
resets per member -- see `configs/experiment/deep_ensemble_*.yaml`), `seed: 12345`,
`test: False`.

## Budget

- **Search runs at a shortened proxy budget**: `max_epochs: 50`, `min_epochs: 10`.
  Every experiment config wires `model.scheduler.T_max: ${trainer.max_epochs}`, so
  `CosineAnnealingLR` rescales to the proxy budget automatically -- no separate
  schedule override is needed here (unlike the old `MultiStepLR` setup, which required
  hand-rescaling `milestones` to match fractions of the budget). This is a real
  advantage of the cosine schedule over a milestone-based one for this protocol.
- Early stopping is the pruner: `monitor: val/auprc`, `mode: max`, `patience: 8`,
  `min_delta: 0.0`. (`hydra-optuna-sweeper==1.2.0` pins optuna 2.x, which exposes no
  Optuna-native pruner -- early stopping is the only mechanism available to kill bad
  trials early.)
- `n_trials: 48`, `n_jobs: 8`, `TPESampler(seed=1234, n_startup_trials=10,
  multivariate=True)`. `n_startup_trials` tracks `n_jobs` (trials per batch), not
  `n_trials` -- setting it equal to `n_trials` (as the old, now-deleted
  `baseline_tang.yaml`/`sngp_tang.yaml` sweep configs did) means TPE never actually
  engages and every trial is pure random search. `multivariate=True` because LR and
  batch size are coupled (the linear/sqrt LR-scaling rule) and, at fixed
  `max_epochs`, a larger batch means fewer gradient updates -- don't read a tuned LR
  in isolation from the batch size it was tuned alongside.
- **Retrain the top-3 trials, not just the winner**, at the experiment's full
  `max_epochs=150`, and pick the final config by full-budget `val/auprc_best`.
  Proxy-budget rankings are noisy; three extra runs per sweep is cheap insurance.
- **Final models**: the winning config x 5 seeds, reported mean +/- std.

## Before trusting any result: measure the noise floor

Run the current default config 5x with different seeds and record the spread of
`val/auprc_best`. If the gap between a sweep's best and 10th-best trial falls inside
that spread, the sweep found nothing distinguishable from seed noise -- report that
plainly rather than picking a spurious winner. Rare-class AUPRC (e.g. Tang's `caa`
class, 1909 samples) will likely dominate this variance.

## Running a sweep

```bash
# One family x one dataset. Family must have a configs/hparams_search/<family>.yaml
# (baseline | sngp); dataset must have a configs/experiment/<family>_<dataset>.yaml.
scripts/hpo/sweep.sh baseline tang
scripts/hpo/sweep.sh sngp     acevedo
```

This submits to SLURM via `hydra/launcher=submitit_slurm`
(`configs/hydra/launcher/submitit_slurm.yaml`), fanning out `n_jobs` concurrent trials
per Optuna batch. Recommended order: run `baseline tang` first as a pilot, inspect its
W&B group's parallel-coordinates plot, and only launch the remaining 7 sweeps
(`{baseline, sngp} x {tang, acevedo, wong, kather2018}`) once that looks sane.

Ad hoc / local (no SLURM), e.g. for a quick smoke test:

```bash
mkdir -p "${EXPERIMENTS_HOME}/${PROJECT_NAME}/optuna"   # sqlite needs this to pre-exist
uv run src/train.py -m hparams_search=baseline experiment=baseline_tang \
  hydra.sweeper.n_trials=2 hydra.sweeper.n_jobs=1
```

Every trial logs to W&B individually, grouped under `group: "${name}_hpo"` /
`job_type: "sweep"` (`log_model: False` -- a 48-trial sweep must not upload 48
checkpoints as W&B artifacts). This gives the same parallel-coordinates and
hyperparameter-importance panels a dedicated W&B Sweep would, without a second
search-space definition living outside the Hydra config tree.

## Reading results

```bash
uv run scripts/hpo/summarize_study.py --study tang_baseline_resnet18_hpo --top-k 3
```

Prints the top-K trials ranked by `val/auprc_best`, each with its parameter overrides
as a ready-to-paste CLI string for the full-budget retrain:

```bash
uv run src/train.py experiment=baseline_tang \
  model.optimizer.lr=0.00034 model.optimizer.weight_decay=0.0021 \
  data.datamodule.batch_size=64 \
  test=True
```

## Sweep-only safety nets

Two behaviors in `src/train.py` are active only when a composed config sets
`sweep_fail_safe: true` (all `hparams_search/*.yaml` do; a normal single run never
does):

- **One bad trial can't abort the study.** `task_wrapper` (`src/utils/utils.py`)
  re-raises on any exception by design, which would otherwise kill an entire Optuna
  study on the first OOM or divergent hyperparameter combination partway through a
  48-trial sweep. `main()` catches that and returns `0.0` (a natural floor for
  AUPRC ∈ [0, 1]) instead, so the sweep continues.
- **A `test/*` metric can never be the objective.** `train()`'s `metric_dict` merges
  train and test metrics into one dict, so `optimized_metric: "test/..."` is one typo
  away from selecting hyperparameters using the test set. `main()` refuses outright if
  `optimized_metric` starts with `test/`.
