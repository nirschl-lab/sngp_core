# Baseline Workflow

The three stages that take a Baseline model from an untuned config to reported
inference results, in order, for one dataset. Same spine as
[SNGP_GUIDE.md](SNGP_GUIDE.md) minus the post-hoc tuning stage -- this family has no
inference-only knob to fit. The *why* behind the selection protocol is
[../HPO_GUIDE.md](../HPO_GUIDE.md); it is not repeated here.

`<dataset>` is whichever `configs/experiment/baseline_<dataset>.yaml` exists for:
`tang`, `acevedo`, `wong`, `kather2018`, `wong_ucdavis`, `wong_upitt`,
`wong_utsouthwestern`.

The tuned config this produces is also what Deep Ensemble uses -- that family is not
swept independently, since its members are baseline models differing only in init seed
(see [DEEP_ENSEMBLES_GUIDE.md](DEEP_ENSEMBLES_GUIDE.md)).

## 1. Sweep

```bash
scripts/hpo/sweep.sh baseline <dataset>
```

48 Optuna trials on SLURM, 8 concurrent, at a shortened proxy budget
(`max_epochs: 50`), selecting on `val/auprc_best`. Searched: `lr`, `weight_decay`,
`batch_size`, `cb_beta`, `focal_gamma`.

`dropout_p` is **not** searched -- see [MC-Dropout at inference](#mc-dropout-at-inference)
below for why that matters.

## 2. Read the study, retrain at full budget

```bash
uv run scripts/hpo/summarize_study.py --study <dataset>_baseline_resnet18_hpo --top-k 3
```

The study name is the experiment's `name` plus `_hpo` -- `acevedo_baseline_resnet18_hpo`
for `baseline_acevedo`. Each trial prints its overrides as a ready-to-paste CLI string.

Retrain the **top-3**, not just the winner (proxy-budget rankings are noisy), at the
experiment's own full `max_epochs: 150`, then pick by full-budget `val/auprc_best`:

```bash
uv run src/train.py experiment=baseline_<dataset> \
    model.optimizer.lr=... model.optimizer.weight_decay=... \
    data.datamodule.batch_size=... \
    test=True
```

Then train the final models -- the winning config × 5 seeds, reported mean ± std.
Never report inference off a sweep checkpoint: those ran the 50-epoch proxy budget,
not the experiment's real one.

## 3. Inference

Confirm the checkpoint's architecture and class count before committing to a long run:

```bash
uv run python -c "
from src.checkpointing.io import read_meta
print(read_meta('<ckpt_path>'))"
```

```bash
uv run src/inference/infer.py \
    ckpt_path=<final>/checkpoints/best.ckpt \
    data=<dataset> \
    fold=test \
    save_path=<out>/baseline/<dataset>
```

Runtime overrides, artifact (paired real/simulated) inference and cross-dataset OOD
sweeps: [../INFERENCE_GUIDE.md](../INFERENCE_GUIDE.md).

Record the checkpoint in [../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md)
and the output directory in
[../MASTER_INFER_RESULTS_PATH.md](../MASTER_INFER_RESULTS_PATH.md).

## MC-Dropout at inference

This is the Baseline family's uncertainty mechanism, and the only one of the three
that is opt-in at inference time:

```bash
uv run src/inference/infer.py \
    ckpt_path=<final>/checkpoints/best.ckpt \
    data=<dataset> \
    infer.runtime.use_mc_dropout=true infer.runtime.mc_passes=20
```

`model.net.dropout_p` is fixed at `0.2` and deliberately never tuned: tuning it against
a *deterministic* validation forward pass would push it toward 0, silently breaking
MC-Dropout without any metric in the sweep noticing. That is why it stays out of the
search space in stage 1.

Per-pass logits, the `uncertainty` column, and the silent no-op against SNGP/Deep
Ensemble checkpoints: [../INFERENCE_GUIDE.md#mc-dropout](../INFERENCE_GUIDE.md#mc-dropout).
