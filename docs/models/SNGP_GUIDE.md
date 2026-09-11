# SNGP Workflow

The four stages that take an SNGP model from an untuned config to reported inference
results, in order, for one dataset. This is a sequence doc -- the *why* behind the
selection protocol is [../HPO_GUIDE.md](../HPO_GUIDE.md), and the *why* behind the GP
head's behavior is
[../DEVELOPMENT.md#sngp-precision-matrix-and-mean-field](../DEVELOPMENT.md#sngp-precision-matrix-and-mean-field).
Neither is repeated here.

`<dataset>` is whichever `configs/experiment/sngp_<dataset>.yaml` exists for: `tang`,
`acevedo`, `wong`, `kather2018`, `wong_ucdavis`, `wong_upitt`, `wong_utsouthwestern`.

## 1. Sweep

```bash
scripts/hpo/sweep.sh sngp <dataset>
```

48 Optuna trials on SLURM, 8 concurrent, at a shortened proxy budget
(`max_epochs: 50`), selecting on `val/auprc_best`. Searched: `lr`, `weight_decay`,
`batch_size`, `cb_beta`, `focal_gamma`, plus SNGP's own `length_scale` and `rff_dim`.

`ridge_penalty` and `mean_field_factor` are deliberately **not** searched -- both are
inference-only under canonical SNGP, so they cannot move the macro-AUPRC objective at
all. See [../HPO_GUIDE.md#guiding-principle](../HPO_GUIDE.md#guiding-principle).

## 2. Read the study, retrain at full budget

```bash
uv run scripts/hpo/summarize_study.py --study <dataset>_sngp_resnet18_hpo --top-k 3
```

The study name is the experiment's `name` plus `_hpo` -- `acevedo_sngp_resnet18_hpo`
for `sngp_acevedo`. Each trial prints its overrides as a ready-to-paste CLI string.

Retrain the **top-3**, not just the winner (proxy-budget rankings are noisy), at the
experiment's own full `max_epochs: 150`, then pick by full-budget `val/auprc_best`:

```bash
uv run src/train.py experiment=sngp_<dataset> \
    model.optimizer.lr=... model.optimizer.weight_decay=... \
    data.datamodule.batch_size=... \
    model.net.length_scale=... model.net.rff_dim=... \
    test=True
```

Never report inference off a sweep checkpoint -- those ran the 50-epoch proxy budget,
not the experiment's real one.

## 3. Tune `mean_field_factor`

```bash
uv run python scripts/checkpoints/tune_sngp_mean_field.py \
    --ckpt <winner>/checkpoints/best.ckpt --experiment sngp_<dataset> --split val
```

One forward pass rather than a training run, because `precision_accum` ships inside
the checkpoint. Select on **validation NLL** -- a proper scoring rule, and unlike ECE
it has no bin-count artifact -- then report ECE on test. `--ridge 1e-3 1.0` sweeps
`ridge_penalty` in the same pass, one extra forward pass per value.

### Applying the tuned value

The script prints a factor; it does not write one. `mean_field_factor` is part of the
net's `spec`, and inference is checkpoint-authoritative --
`configs/infer/runtime/default.yaml` deliberately exposes no architecture override --
so **a tuned factor only reaches inference if it was in the config when the checkpoint
was written**. Add it to the existing `model.net` block in
`configs/experiment/sngp_<dataset>.yaml` before the final runs -- it is absent there
today, so every experiment currently inherits the `1.0` default from
`configs/model/sngp_classifier.yaml`:

```yaml
model:
  net:
    arch: "resnet18"
    # ... existing length_scale / rff_dim / ridge_penalty ...
    mean_field_factor: <tuned value>
```

then train the final models -- the winning config × 5 seeds, reported mean ± std.

This is an ordering constraint, not a compute cost: the factor has zero effect on the
training fit (the CE loss sees raw logits), so the weights are exactly what they would
have been. Skipping it silently runs inference at the `1.0` default.

## 4. Inference

Confirm the checkpoint carries what you think it does before committing to a long run:

```bash
uv run python -c "
from src.checkpointing.io import read_meta
print(read_meta('<ckpt_path>'))"
```

`net_spec` should show the tuned `mean_field_factor`, and `num_classes` should match
the dataset you are about to run on. Then:

```bash
uv run src/inference/infer.py \
    ckpt_path=<final>/checkpoints/best.ckpt \
    data=<dataset> \
    fold=test \
    save_path=<out>/sngp/<dataset>
```

Runtime overrides, artifact (paired real/simulated) inference and cross-dataset OOD
sweeps: [../INFERENCE_GUIDE.md](../INFERENCE_GUIDE.md).

Record the checkpoint in [../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md)
and the output directory in
[../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md).

## Gotchas

- **Pre-correction checkpoints do not load.** The old head stored `cov_ema`/
  `num_updates` instead of `precision_accum`/`covariance`, and its weights were trained
  against mean-field-corrected logits. Anything from before the canonical-SNGP
  correction -- including a sweep whose study database you renamed `old_*` -- is
  off-regime and has to be re-run. `load_state_dict` raises pointing at the
  `sngp-pre-correction` tag.
- **resnet backbones only.** ViT is rejected at construction -- see
  [../SUPPORTED_MODELS.md#sngp--vit-compatibility](../SUPPORTED_MODELS.md#sngp--vit-compatibility).
- **Ad-hoc local sweeps need the sqlite directory to pre-exist**
  (`mkdir -p "${EXPERIMENTS_HOME}/${PROJECT_NAME}/optuna"`); `sweep.sh` does this for
  you, a bare `-m hparams_search=sngp` invocation does not.
- **`val/nll` and `val/ece` never feed selection.** If the AUPRC-optimal config comes
  out materially worse calibrated than the untuned default, that is a finding worth
  reporting, not a reason to change the objective.
