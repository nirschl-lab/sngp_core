# SNGP Workflow

The four stages that take an SNGP model from an untuned config to reported inference
results, in order, for one dataset. This is a sequence doc -- the *why* behind the
selection protocol is [../HPO_GUIDE.md](../HPO_GUIDE.md), and the *why* behind the GP
head's behavior is
[../DEVELOPMENT.md#sngp-precision-matrix-and-mean-field](../DEVELOPMENT.md#sngp-precision-matrix-and-mean-field).
Neither is repeated here.

`<dataset>` is whichever `configs/experiment/sngp_<dataset>.yaml` exists for: `tang`,
`acevedo`, `wong`, `kather2018`, `wong_ucdavis`, `wong_upitt`, `wong_utsouthwestern`.

`sngp_cifar100` also exists but is **not** part of this workflow: it reproduces the SNGP
paper's own CIFAR-100 / WideResNet-28-10 benchmark and is deliberately off-protocol
(SGD + piecewise schedule, `val/loss` selection, no sweep, no post-hoc calibration). It
has its own page, [CIFAR100_BENCHMARK.md](CIFAR100_BENCHMARK.md).

Every SNGP constant (`rff_dim: 1024`, `length_scale: 1.4142`, `ridge_penalty`,
`cov_momentum`, ...) is fixed to Liu et al. (2022) Table 9 in
`configs/model/sngp_classifier.yaml`; the experiment configs set only `arch` /
`num_classes` / `pretrained`, and `tests/test_configs.py` rejects anything else. Two
knobs are not constants and are exempt from that check: `spectral_norm_bound` (the
paper's `c`, swept in stage 1 -- each dataset pins its own winner in its experiment
config, and `configs/model/sngp_classifier.yaml`'s `6.0` is the default for datasets
that have not been re-swept) and `mean_field_factor` (the paper's kernel amplitude
sigma, fit in stage 3).

## 1. Sweep

```bash
scripts/hpo/sweep.sh sngp <dataset>
```

48 W&B Bayesian trials on SLURM, 8 concurrent, at the proxy budget (`max_epochs: 50`),
minimizing `val/nll_cal_best` -- validation NLL after refitting the mean-field factor on
each epoch's validation outputs. Searched: `lr`, `weight_decay`, `batch_size`, and
`model.net.spectral_norm_bound` over `{1, 2, 4, 6, 8}`.

`ridge_penalty` and `mean_field_factor` are deliberately **not** searched -- both are
inference-only under canonical SNGP, so they cannot move the training fit; they are fit
post-hoc in stage 3. `rff_dim` / `length_scale` are fixed to the paper.

Pilot the launcher on a tiny run first, and watch how `c` reads off the sweep: the paper's
rule is the *smallest* `c` that keeps accuracy, which an NLL-minimizing search will not
prefer on its own -- see [../HPO_GUIDE.md#search-space](../HPO_GUIDE.md#search-space).

## 2. Read the sweep, retrain at full budget

```bash
uv run scripts/hpo/summarize_sweep.py --sweep <entity>/<project>/<sweep_id> --top-k 3
```

The sweep path is in [../MASTER_SWEEPS.md](../MASTER_SWEEPS.md) (its name is the
experiment's `name` plus `_hpo`, e.g. `acevedo_sngp_resnet18_hpo`). Each trial prints its
overrides as a ready-to-paste CLI string.

Retrain the **top-3**, not just the winner (proxy-budget rankings are noisy), at the
experiment's own full `max_epochs: 150`, then pick by full-budget `val/nll_cal_best`:

```bash
uv run src/train.py experiment=sngp_<dataset> \
    model.optimizer.lr=... model.optimizer.weight_decay=... \
    data.datamodule.batch_size=... model.net.spectral_norm_bound=... \
    test=True
```

Then train the final models -- the winning config x 5 seeds -- and write the winning
values into `configs/experiment/sngp_<dataset>.yaml` (replacing the `# was ...` placeholders)
so the experiment file *is* the protocol for that dataset. Never report inference off a
sweep checkpoint -- those ran the 50-epoch proxy budget, not the experiment's real one.

## 3. Calibrate: fit the mean-field factor (sigma)

```bash
uv run scripts/checkpoints/calibrate_checkpoint.py \
    --ckpt <final>/checkpoints/best.ckpt --experiment sngp_<dataset> --split val
# -> <final>/checkpoints/best.calibrated.ckpt
```

One forward pass rather than a training run, because `precision_accum` ships inside the
checkpoint. The script fits `mean_field_factor` by minimizing **validation NLL** -- a
proper scoring rule, and unlike ECE it has no bin-count artifact -- prints the grid
table (NLL, smooth-ECE, accuracy, macro-F1, macro-AUPRC; accuracy and F1 must be
identical on every row, and the script asserts the argmax is unchanged), and writes a
sibling `best.calibrated.ckpt` whose `net_spec.mean_field_factor` is the fitted value,
with the provenance under `sngp_core.calibration`. `--ridge 1e-3 1.0` also revisits
`ridge_penalty` in the same pass (one extra forward pass per value) and writes the winner.

Nothing else to do: inference is checkpoint-authoritative, so the calibrated checkpoint
carries its own factor. The source `best.ckpt` is never modified. To report the fitted
factor on the test split afterwards:

```bash
uv run scripts/checkpoints/calibrate_checkpoint.py \
    --ckpt <final>/checkpoints/best.calibrated.ckpt --experiment sngp_<dataset> \
    --split test --report-only
```

This is SNGP's counterpart of the Baseline's temperature (stage 3 of
[BASELINE_GUIDE.md](BASELINE_GUIDE.md)): both families get exactly one post-hoc scalar,
fit the same way on the same split.

## 4. Inference

Confirm the checkpoint carries what you think it does before committing to a long run:

```bash
uv run python -c "
from src.checkpointing.io import read_meta
print(read_meta('<ckpt_path>'))"
```

`net_spec` should show the swept `spectral_norm_bound` and the fitted
`mean_field_factor`, `calibration` should be non-`None` (knob, split, NLL before/after),
and `num_classes` should match the dataset you are about to run on. Then:

```bash
uv run src/inference/infer.py \
    ckpt_path=<final>/checkpoints/best.calibrated.ckpt \
    data=<dataset> \
    fold=test \
    save_path=<out>/sngp/<dataset>
```

The default output folder gets a `_calibrated` suffix so it cannot overwrite an
uncalibrated run of the same checkpoint. Runtime overrides, artifact (paired
real/simulated) inference and cross-dataset OOD sweeps:
[../INFERENCE_GUIDE.md](../INFERENCE_GUIDE.md).

Record both checkpoints in [../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md)
(`best.ckpt` = trained, `best.calibrated.ckpt` = report from this) and the output
directory in [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md).

## The Laplace weight (`likelihood`)

The posterior precision is `P = ridge*I + sum_i w_i phi_i phi_i^T`. For a multinomial
logistic likelihood the per-example Hessian is `diag(p) - p p^T`, a `[K, K]` matrix — but
`precision_accum` is a single `[rff_dim, rff_dim]`, so that Hessian has to be reduced to
one scalar `w_i`. `likelihood` picks the reduction:

| value | `w_i` | notes |
|---|---|---|
| `gaussian` (default) | `1` | Unit weight. The reference default for image classification and this project's protocol constant. |
| `binary_logistic` | `p_max (1 - p_max)` | Reads the top class as a one-vs-rest Bernoulli and discards the rest — edward2's own reduction. `<= 0.25`. |
| `trace_logistic` | `1 - \|\|p\|\|^2 = sum_k p_k (1 - p_k)` | The trace of the full Hessian: uses every class, costs nothing extra, and is `>=` the `binary_logistic` weight. `<= 1 - 1/K`. |

`per_class_logistic` — the unreduced `[B, K]` diagonal `p (1 - p)` — is implemented in
`RandomFeatureGaussianProcess._laplace_weights` but **refused at construction**. It would
need a `[K, rff_dim, rff_dim]` accumulator (419 MB per buffer at `K=100, rff_dim=1024`) and
would return a `[B, K]` predictive variance, which neither the `[B, 1]`
`ModelOutput.variance` contract nor the one-scalar-per-row `predictions.csv` schema carries.

Two properties worth holding on to:

- **It never touches training.** The weight only feeds the precision accumulator; the loss
  sees raw logits and no variance is computed in train mode. Two runs differing only in
  `likelihood` train the *same* model — what differs is the eval-time predictive variance,
  and so the mean-field-corrected logits. `tests/models/sngp/test_rff.py::TestLikelihood::
  test_likelihood_does_not_touch_the_training_signal` pins this.
- **It interacts with `mean_field_factor`.** A logistic weight shrinks the accumulator, which
  inflates the variance, which strengthens the mean-field correction at a fixed factor. On a
  converged classifier the effect is large (`w ~ 0.02` at CIFAR-100 train accuracy), so a
  factor fitted for `gaussian` will badly over-shrink a logistic run. **Refit
  `mean_field_factor` (§3) after changing `likelihood`** — or, for a pinned-factor benchmark,
  sweep it offline before comparing arms.

## Gotchas

- **Pre-correction checkpoints do not load.** The old head stored `cov_ema`/
  `num_updates` instead of `precision_accum`/`covariance`, and its weights were trained
  against mean-field-corrected logits. Anything from before the canonical-SNGP
  correction is off-regime and has to be re-run. `load_state_dict` raises pointing at the
  `sngp-pre-correction` tag.
- **Checkpoints from before the spectral-norm bound existed still load** and rebuild
  identically: their spec lacks `spectral_norm_bound`, and the ctor default `None` is the
  stock hard normalization they were trained with. They are off-protocol for the
  comparison (hard normalization, focal loss), not broken.
- **resnet backbones only.** ViT is rejected at construction -- see
  [../SUPPORTED_MODELS.md#sngp--vit-compatibility](../SUPPORTED_MODELS.md#sngp--vit-compatibility).
- **`c` is measured on the reshaped-matrix spectral norm**, not the conv operator norm
  edward2 uses, so the paper's `c = 6` is a hint; extend the grid if the sweep's optimum
  sits on an edge.
- **`eval.py` is config-authoritative.** Evaluating a calibrated checkpoint through
  `src/eval.py` needs `model.net.mean_field_factor=<fitted>` on the command line, or it
  refuses with a spec mismatch; `src/inference/infer.py` reads the checkpoint and needs
  nothing.
