# Hyperparameter Search Guide

How to tune each model family fairly before comparing Baseline, SNGP, and Deep
Ensemble against each other. The protocol exists because a comparison is only as fair
as the tuning behind it: every family gets the same backbone, optimizer family,
schedule, loss, budget, objective, and the same one-scalar post-hoc calibration freedom.
It follows Liu et al. (2022, JMLR, *A Simple Approach to Improve Single-Model Deep
Uncertainty via Distance-Awareness*) for what is fixed and what is swept.

## Guiding principle

**Select on calibrated validation NLL, then calibrate post-hoc, then report.**

The objective is `val/nll_cal_best`: the running minimum, across epochs, of validation
NLL *after* fitting the family's single post-hoc calibration knob on that epoch's
validation outputs (`LitModuleBase`, `src/metrics/posthoc_calibration.py`). Each family
has exactly one such knob, and it is inference-only -- it cannot move the training fit,
accuracy, or macro-F1 (it divides every logit of an example by one positive scalar):

| Family | Knob | Applied as | Origin |
|---|---|---|---|
| Baseline, Deep Ensemble | `temperature` | `logits / T` | temperature scaling (Guo et al. 2017) |
| SNGP | `mean_field_factor` | `raw_logits / sqrt(1 + factor * variance)` | the paper's kernel amplitude sigma; the reference implementation collapses it into `gp_mean_field_factor`, and the paper estimates it on held-out data by minimizing the log score |

Why the *calibrated* NLL rather than the raw one: it is the quantity a post-hoc-calibrated
final model actually reports, and raw NLL rises with late-training overconfidence in a
way that differs between families (SNGP carries a partial correction at factor 1.0,
Baseline none), so selecting on raw NLL would not compare like with like. NLL is a
strictly proper scoring rule; ECE is not (it has a bin-count artifact and a trivial
minimizer), which is why NLL is the objective and ECE is reported. `val/nll` /
`val/nll_best` (raw), the fitted knob (`val/temperature_fit` /
`val/mean_field_factor_fit`), `val/auprc*` and `val/ece` are logged alongside as
diagnostics. Report accuracy, macro-F1 and macro-AUPRC next to NLL/ECE: if NLL
selection costs discriminative performance, that is a finding to show, not hide.

This replaces the earlier rule that calibration must never be a selection axis. The
consequence for the paper's narrative: the claim becomes "with equal tuning budget and
one post-hoc knob each, family X reaches lower NLL/ECE", not "family X is better
calibrated out of the box".

## Loss: plain cross-entropy, everywhere

Every family and dataset trains with unweighted `CrossEntropyLoss`
(`configs/model/*_classifier.yaml`: `class_freq: null`, `class_weights: null`,
`label_smoothing: 0.0`). Focal / class-balanced reweighting reshapes calibration
(Mukhoti et al. 2020, *Calibrating Deep Neural Networks using Focal Loss*), so tuning
`cb_beta` / `focal_gamma` per family would confound the very comparison this project
makes. Imbalance is handled at **reporting** time (macro metrics, per-class recall),
never in the loss:

| Dataset | Train-split `class_freq` | Max : min |
|---|---|---|
| tang | 1909, 2428, 47702, 9331 | ~25 : 1 |
| acevedo | 860, 2176, 1086, 2032, 842, 987, 2339, 1642 | ~2.8 : 1 |
| kather2018 | 10039, 9464, 7274, 8126, 8100, 6109, 7281, 7398, 6209 | ~1.6 : 1 |
| wong | 12301, 12288, 12223, 12218 | balanced |

Watch Tang's minority classes (`caa`, 1909 samples) on the first Tang sweep: a recall
collapse under plain CE is a finding, not a bug. `class_weights` remains as the documented
fallback (weighted CE), but weighting breaks NLL's proper-scoring-rule property; if it is
ever used it must be identical across families for that dataset --
`tests/test_configs.py::TestExperimentProtocolConsistency` enforces this, and the
`class_freq` values stay in `configs/data/*.yaml` as provenance only.

## Selection metric

`val/nll_cal_best` -- a `torchmetrics.MinMetric` over `val/nll_cal`, computed in
`LitModuleBase.on_validation_epoch_end`. The knob choice is structural, not by family
name: if the net reports `raw_logits` and `variance` (SNGP in eval mode) the mean-field
factor is fit, otherwise a temperature. `model.calibration_knob=temperature` forces a
temperature fit on SNGP for a like-for-like ablation.

Why a running best: `train()`'s `metric_dict` (`src/train.py`) is a snapshot of the
*last* epoch, and W&B's sweep controller reads the run **summary**, which is also the last
logged value. A running minimum makes "last" equal "best" for both readers. The
epoch-level `val/nll_cal` is what `EarlyStopping` and `ModelCheckpoint` monitor -- in
sweeps (`configs/hparams_search/<family>.yaml`) **and** in final runs
(`configs/callbacks/default.yaml`), so `best.ckpt` is the epoch the objective picked.

## Search space

Single source of truth: `configs/hparams_search/wandb/<family>.yaml` (native W&B sweep
YAML). The Hydra side, `configs/hparams_search/<family>.yaml`, is a *run preset* (proxy
budget, objective, monitors, logger) with no search space of its own;
`tests/test_sweeps.py::TestWandbSweepConfigDrift` asserts every swept key resolves in
the composed config and that `metric.name` equals the preset's `optimized_metric`.

Shared by both swept families:

| Param | Distribution | Protocol default |
|---|---|---|
| `model.optimizer.lr` | log-uniform 1e-5 .. 1e-2 | 1e-3 |
| `model.optimizer.weight_decay` | log-uniform 1e-6 .. 1e-2 | 1e-4 |
| `data.datamodule.batch_size` | {64, 128, 256} | 128 |

SNGP only:

| Param | Values | Protocol default |
|---|---|---|
| `model.net.spectral_norm_bound` | {1, 2, 4, 6, 8} | 6.0 |

`spectral_norm_bound` is the paper's `c` (eq. 15: `W <- c * W / sigma` only when
`sigma > c`). It trades the residual blocks' expressiveness against their distance
preservation and *changes the trained function*, which is why it is the one SNGP knob in
the sweep. Two cautions when reading it off: (1) an NLL-minimizing sweep has no reason to
prefer a small `c`, but small `c` is what buys distance-awareness -- the paper's rule is
"the smallest `c` that keeps accuracy", so read the parallel-coordinates plot for `c` and
apply that rule if NLL is flat across the grid; (2) `c` is measured on torch's
reshaped-matrix spectral norm, not the conv operator norm edward2 estimates, so the
paper's `c = 6` is a hint, not a transferable value -- extend the grid if the optimum sits
on an edge.

Deep Ensemble is **not swept**: members are baseline nets differing only in init seed,
so it inherits the tuned Baseline config (`tests/test_configs.py` checks the two agree).

## Fixed protocol constants

Never in the search space; live once in `configs/model/*_classifier.yaml` (experiments
set only `arch` / `num_classes` / `pretrained`, and the config tests reject overrides):

| Constant | Value | Source |
|---|---|---|
| backbone | `resnet18`, `pretrained: false` | shared |
| loss | plain CE, `label_smoothing: 0.0` | above |
| schedule | `CosineAnnealingLR(T_max=max_epochs)` (`CosineAnnealingWarmRestarts` per member for DE) | shared |
| `seed` | 12345 | shared |
| `dropout_p` (Baseline / DE members) | 0.2 | MC-Dropout at inference needs it nonzero; tuning it against a deterministic val pass would push it to 0 |
| `temperature` (Baseline / DE) | 1.0 at train time | post-hoc knob |
| SNGP `rff_dim` | 1024 | Liu et al. Table 9 |
| SNGP `length_scale` | 1.4142 | Table 9 says 2.0 = edward2 `gp_kernel_scale`, which scales the GP input by `1/sqrt(2)`; this code divides `W` by `length_scale`, so `sqrt(2)` is the same kernel width |
| SNGP `mean_field_factor` | 1.0 at train time | post-hoc knob (= sigma) |
| SNGP `ridge_penalty` | 1e-3 | inference-only; `calibrate_checkpoint.py --ridge` can revisit it |
| SNGP `cov_momentum` | -1 (exact per-epoch sum) | reference |
| SNGP `normalize_input` / `likelihood` / `output_bias` / `random_feature_type` / `n_power_iterations_sn` | true / gaussian / false / orf / 1 | reference |

`likelihood` is fixed, not swept, but it has two off-protocol alternatives to `gaussian`
(`binary_logistic`, `trace_logistic` — see
[models/SNGP_GUIDE.md](models/SNGP_GUIDE.md#the-laplace-weight-likelihood)). It is
inference-only in the strictest sense: it feeds the precision accumulator and never the
loss, so two runs differing only in `likelihood` train the same model. That makes it a
poor sweep target — changing it is better done as a named arm than as an HPO dimension.

`SNGPClassifier`'s constructor defaults equal these (so `deep_ensemble_sngp_*` members,
built from the spec, get them too), except `spectral_norm_bound`, whose ctor default stays
`None` (stock hard normalization) so checkpoints written before the bound existed rebuild
bit-identically. `spectral_norm_bound` is swept, not fixed, so it is absent from the table
above: `configs/model/sngp_classifier.yaml` states `6.0` as the family default, and a
dataset whose sweep picked something else pins that winner in
`configs/experiment/sngp_<dataset>.yaml` alongside its `lr`/`weight_decay` (e.g.
`sngp_acevedo` pins `4.0`). An SNGP ensemble's `base_model_spec.spectral_norm_bound` must
match its partner SNGP experiment's value (`tests/test_configs.py`).

## Budget

- **Proxy budget**: `max_epochs: 50`, `min_epochs: 10`. `model.scheduler.T_max` is wired
  to `${trainer.max_epochs}` in every experiment config, so the cosine schedule rescales
  itself.
- **Trials**: `run_cap: 48`, up to 8 concurrent (`sbatch --array=0-47%8`, one trial per
  array task by default: a 3 h limit and a SLURM log per trial, crash isolation).
  `method: bayes`.
- **Pruning**: W&B hyperband (`min_iter: 10`, `eta: 3`) counts one iteration per logged
  value of the objective, and `LitModuleBase` logs it once per epoch, so brackets are at
  10 and 30 epochs; Lightning's own `EarlyStopping(monitor=val/nll_cal, patience=8,
  min_delta=0.0)` runs underneath. Delete the `early_terminate` block to rely on early
  stopping alone.
- **Retrain the top-3 trials, not just the winner**, at the experiment's full
  `max_epochs=150`, and pick the final config by full-budget `val/nll_cal_best`.
  Proxy-budget rankings are noisy; three extra runs per sweep is cheap insurance.
  This is a *separate manual step*, not something the sweep or `summarize_sweep.py`
  does -- the trials themselves only ever ran at 50 epochs, and they leave no
  checkpoint behind (`save_top_k: 0`), so retraining is mandatory rather than an
  optimization. See [Reading results](#reading-results) for the exact command.
- **Final models**: the winning config x 5 seeds, reported mean +/- std, each
  post-hoc calibrated (next section).

## Before trusting any result: measure the noise floor

Run the protocol-default config 5x with different seeds and record the spread of
`val/nll_cal_best`. If the gap between a sweep's best and 10th-best trial falls inside
that spread, the sweep found nothing distinguishable from seed noise -- report that
plainly rather than picking a spurious winner.

## Prerequisites

- `wandb login` once on the machine that creates sweeps and on the cluster login node.
- `.env` (see `env_example`) must set `EXPERIMENTS_HOME` and `PROJECT_NAME`; the W&B
  project **is** `PROJECT_NAME`, the same one every experiment logs to. `WANDB_ENTITY` is
  optional (defaults to your W&B default entity).
- Nothing else: there is no sqlite study directory and no sqlalchemy pin any more --
  `hydra-optuna-sweeper` is gone.

## Running a sweep

```bash
# One family x one dataset: creates the sweep, appends it to docs/MASTER_SWEEPS.md,
# and submits a SLURM array of `wandb agent`s (scripts/slurm/wandb_agent.sbatch).
scripts/hpo/sweep.sh baseline tang
scripts/hpo/sweep.sh sngp     acevedo --trials 48 --parallel 8

# Pilot first: two tiny trials, not registered.
scripts/hpo/sweep.sh baseline tang --trials 2 --parallel 2 --no-register \
    --override trainer.max_epochs=2 --override +trainer.limit_train_batches=0.05

# Create only, run an agent by hand (e.g. on a workstation GPU):
scripts/hpo/sweep.sh sngp acevedo --no-submit
uv run wandb agent --count 1 <entity>/<project>/<sweep_id>

# Show the final sweep YAML without touching W&B:
scripts/hpo/sweep.sh sngp acevedo --dry-run
```

Recommended order: pilot `baseline tang`, check in the W&B UI that the trials sit
*inside* the sweep (grouped `tang_baseline_resnet18_hpo`, named by W&B) and that
`val/nll_cal_best` shows up in the sweep's parallel-coordinates panel, then launch the
remaining `{baseline, sngp} x {tang, acevedo, wong, kather2018}`.

Each trial is an ordinary single run: `src/train.py hparams_search=<family>
experiment=<family>_<dataset> <overrides>`. Trial outputs land under
`train/<model.name>_<data.name>/sweeps/<wandb_sweep_id>/<timestamp>_<wandb_run_id>/`
(docs/OUTPUT_LAYOUT.md) -- the sweep id is the grouping level, so a pilot and the real run stay
apart; look it up in [MASTER_SWEEPS.md](MASTER_SWEEPS.md). `log_model: False` keeps 48
checkpoints out of W&B artifacts, and
`save_top_k: 0` / `save_last: False` keeps them off local disk too -- a trial dir is ~200 KB
of logs and config, not 278 MB of weights nothing ever loads.
A crashed trial is just a crashed W&B run that the Bayesian search ignores -- there is
deliberately no fail-safe that swallows exceptions and returns a floor value (under a
minimized objective such a floor would rank as the *best* trial).

## Reading results

```bash
uv run scripts/hpo/summarize_sweep.py --sweep <entity>/<project>/<sweep_id> --top-k 3
```

`summarize_sweep.py` is a read-only W&B API consumer: it ranks trials that already ran
and launches nothing. **The top-3 it prints are the proxy-budget trials themselves --
50 epochs, not retrained.** Every one of them ran with `hparams_search=<family>` in the
command, which pins `trainer.max_epochs: 50` over the experiment's 150.

It prints each trial with its fitted knob and raw NLL, plus its swept parameters alone
(no `experiment=`, no `hparams_search=`) as a ready-to-paste string:

```
#1  run=lively-sweep-12  id=cq95tpch  val/nll_cal_best=0.4131  (val/nll_best=0.5522, ...)
    data.datamodule.batch_size=64 model.net.spectral_norm_bound=4.0 model.optimizer.lr=0.00034 model.optimizer.weight_decay=0.0021
```

Paste that onto a plain training run -- **once per top-3 config** -- to do the full-budget
retrain:

```bash
uv run src/train.py experiment=sngp_acevedo \
  model.optimizer.lr=0.00034 model.optimizer.weight_decay=0.0021 \
  data.datamodule.batch_size=64 model.net.spectral_norm_bound=4.0 \
  test=True
```

Leaving `hparams_search=` off is the whole mechanism: `trainer.max_epochs` falls back to
the experiment's 150, `model.scheduler.T_max` rescales with it, `EarlyStopping.patience`
returns to 20, and the run lands in `runs/` with a real `best.ckpt` instead of in
`sweeps/` with none. Then compare the three full-budget `val/nll_cal_best` values and
pick the winner -- that is the config you run x 5 seeds.

Sweep paths are recorded in [MASTER_SWEEPS.md](MASTER_SWEEPS.md) by `sweep.py`.

## After the retrain: fit the post-hoc knob, then report

```bash
uv run scripts/checkpoints/calibrate_checkpoint.py \
    --ckpt <final>/checkpoints/best.ckpt --experiment <family>_<dataset> --split val
# -> <final>/checkpoints/best.calibrated.ckpt
```

One forward pass over the validation split fits the family's knob (temperature, or the
mean-field factor -- and `--ridge 1e-3 1.0` also revisits `ridge_penalty` for SNGP) by
minimizing NLL, checks that the argmax is untouched, and writes a **sibling checkpoint**
whose `net_spec` carries the fitted value plus a `sngp_core.calibration` provenance
block (`src/checkpointing/io.py::write_checkpoint_with_net_spec`). No retraining, no
config edit: inference is checkpoint-authoritative, so `src/inference/infer.py
ckpt_path=<...>/best.calibrated.ckpt` uses it, `read_meta` shows it, and the default
inference folder gets a `_calibrated` suffix so it cannot overwrite the uncalibrated run.
Fit on `val`, report on `test` -- the script refuses to fit on the test split
(`--split test` is allowed only with `--report-only`).

Stage-by-stage: [models/BASELINE_GUIDE.md](models/BASELINE_GUIDE.md),
[models/SNGP_GUIDE.md](models/SNGP_GUIDE.md),
[models/DEEP_ENSEMBLES_GUIDE.md](models/DEEP_ENSEMBLES_GUIDE.md).

## Safety nets

- **A `test/*` metric can never be the objective.** `train()`'s `metric_dict` merges
  train and test metrics into one dict, so `optimized_metric: "test/..."` is one typo
  away from selecting hyperparameters on the test set. `src/train.py` refuses outright,
  and the drift test requires `metric.name` to start with `val/` and end in `_best`.
- **The two sweep files cannot drift apart.** `tests/test_sweeps.py` composes each
  preset and checks every swept key resolves, the objective matches, both monitors are
  the epoch-level objective with `mode: min`, the preset is a plain single run (no
  `MULTIRUN`, no fail-safe, `logger.wandb.name: null`), and hyperband cannot kill a trial
  before Lightning's `min_epochs`.
- **Experiments cannot silently re-tune a constant.**
  `tests/test_configs.py::TestExperimentProtocolConsistency` checks the loss config is
  identical across families per dataset and plain CE, SNGP experiments do not override
  the fixed constants, ctor defaults match the model config, both callbacks monitor
  `val/nll_cal`, and Deep Ensemble tracks its partner's optimizer.
