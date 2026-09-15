# Baseline Workflow

The four stages that take a Baseline model from an untuned config to reported
inference results, in order, for one dataset. Same spine as
[SNGP_GUIDE.md](SNGP_GUIDE.md): sweep, retrain, fit the family's one post-hoc
calibration knob (here a temperature), infer. The *why* behind the selection protocol is
[../HPO_GUIDE.md](../HPO_GUIDE.md); it is not repeated here.

`<dataset>` is whichever `configs/experiment/baseline_<dataset>.yaml` exists for:
`tang`, `acevedo`, `wong`, `kather2018`, `wong_ucdavis`, `wong_upitt`,
`wong_utsouthwestern`.

The tuned config this produces is also what Deep Ensemble uses -- that family is not
swept independently, since its members are baseline models differing only in init seed
(see [DEEP_ENSEMBLES_GUIDE.md](DEEP_ENSEMBLES_GUIDE.md)); `tests/test_configs.py` checks
the two experiment files agree.

## 1. Sweep

```bash
scripts/hpo/sweep.sh baseline <dataset>
```

48 W&B Bayesian trials on SLURM, 8 concurrent, at the proxy budget (`max_epochs: 50`),
minimizing `val/nll_cal_best` -- validation NLL after refitting a temperature on each
epoch's validation outputs. Searched: `lr`, `weight_decay`, `batch_size`.

`dropout_p` is **not** searched -- see [MC-Dropout at inference](#mc-dropout-at-inference)
below for why that matters. The loss is plain cross-entropy (no `cb_beta` /
`focal_gamma`; [../HPO_GUIDE.md#loss-plain-cross-entropy-everywhere](../HPO_GUIDE.md#loss-plain-cross-entropy-everywhere)).

## 2. Read the sweep, retrain at full budget

```bash
uv run scripts/hpo/summarize_sweep.py --sweep <entity>/<project>/<sweep_id> --top-k 3
```

The sweep path is in [../MASTER_SWEEPS.md](../MASTER_SWEEPS.md) (its name is the
experiment's `name` plus `_hpo`, e.g. `acevedo_baseline_resnet18_hpo`). Each trial
prints its overrides as a ready-to-paste CLI string.

Retrain the **top-3**, not just the winner (proxy-budget rankings are noisy), at the
experiment's own full `max_epochs: 150`, then pick by full-budget `val/nll_cal_best`:

```bash
uv run src/train.py experiment=baseline_<dataset> \
    model.optimizer.lr=... model.optimizer.weight_decay=... \
    data.datamodule.batch_size=... \
    test=True
```

Then train the final models -- the winning config x 5 seeds -- and write the winning
values into `configs/experiment/baseline_<dataset>.yaml` (replacing the `# was ...`
placeholders) **and** into the matching `deep_ensemble_<dataset>.yaml`. Never report
inference off a sweep checkpoint: those ran the 50-epoch proxy budget, not the
experiment's real one.

## 3. Calibrate: fit the temperature

```bash
uv run scripts/checkpoints/calibrate_checkpoint.py \
    --ckpt <final>/checkpoints/best.ckpt --experiment baseline_<dataset> --split val
# -> <final>/checkpoints/best.calibrated.ckpt
```

One forward pass over the validation split fits `temperature` (`logits / T`, Guo et al.
2017) by minimizing **validation NLL**, prints the grid table (accuracy and macro-F1 are
identical on every row -- a temperature cannot change the argmax, and the script asserts
it), and writes a sibling `best.calibrated.ckpt` whose `net_spec.temperature` is the
fitted value, with the provenance under `sngp_core.calibration`. The source `best.ckpt`
is never modified, and no config edit or retraining is needed: `BaselineClassifier`
applies the temperature from its spec, so inference (checkpoint-authoritative) uses it.

To report the fitted temperature on the test split afterwards:

```bash
uv run scripts/checkpoints/calibrate_checkpoint.py \
    --ckpt <final>/checkpoints/best.calibrated.ckpt --experiment baseline_<dataset> \
    --split test --report-only
```

This is the counterpart of SNGP's `mean_field_factor` (stage 3 of
[SNGP_GUIDE.md](SNGP_GUIDE.md)): both families get exactly one post-hoc scalar, fit the
same way on the same split.

## 4. Inference

Confirm the checkpoint's architecture, class count and calibration before committing to
a long run:

```bash
uv run python -c "
from src.checkpointing.io import read_meta
print(read_meta('<ckpt_path>'))"
```

```bash
uv run src/inference/infer.py \
    ckpt_path=<final>/checkpoints/best.calibrated.ckpt \
    data=<dataset> \
    fold=test \
    save_path=<out>/baseline/<dataset>
```

The default output folder gets a `_calibrated` suffix so it cannot overwrite an
uncalibrated run of the same checkpoint. Runtime overrides, artifact (paired
real/simulated) inference and cross-dataset OOD sweeps:
[../INFERENCE_GUIDE.md](../INFERENCE_GUIDE.md).

Record both checkpoints in [../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md)
(`best.ckpt` = trained, `best.calibrated.ckpt` = report from this) and the output
directory in [../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md).

## MC-Dropout at inference

This is the Baseline family's uncertainty mechanism, and the only one of the three
that is opt-in at inference time:

```bash
uv run src/inference/infer.py \
    ckpt_path=<final>/checkpoints/best.calibrated.ckpt \
    data=<dataset> \
    infer.runtime.use_mc_dropout=true infer.runtime.mc_passes=20
```

`model.net.dropout_p` is fixed at `0.2` and deliberately never tuned: tuning it against
a *deterministic* validation forward pass would push it toward 0, silently breaking
MC-Dropout without any metric in the sweep noticing. That is why it stays out of the
search space in stage 1.

The fitted temperature applies to every MC pass too (`BaselineClassifier.forward` is the
single code path), but it was fit on the *deterministic* forward, so it is approximately,
not exactly, NLL-optimal for the mean-of-softmax MC predictive.

Per-pass logits, the `uncertainty` column, and the silent no-op against SNGP/Deep
Ensemble checkpoints: [../INFERENCE_GUIDE.md#mc-dropout](../INFERENCE_GUIDE.md#mc-dropout).
