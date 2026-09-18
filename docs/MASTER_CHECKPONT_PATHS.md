> **SNGP checkpoints below predate the canonical-SNGP correction.** They were trained
> with the old GP head (mean-field applied inside the training loss, `cov_ema` instead of
> a precision matrix) and will **not** load on the `sngp-corrections` branch or later --
> `load_state_dict` raises with a pointer to the `sngp-pre-correction` tag. Check out that
> tag, or the `isbi2026` branch, to use them; they remain the ISBI 2026 paper's
> checkpoints. Baseline and Deep-Ensemble checkpoints are unaffected.

> **Convention for new entries.** Each final run has two checkpoints in the same
> `checkpoints/` directory: `best.ckpt` (straight out of training) and
> `best.calibrated.ckpt` (written by `scripts/checkpoints/calibrate_checkpoint.py` with the
> post-hoc knob -- `temperature` or `mean_field_factor` -- fit on the validation split).
> Record both; **report inference from `best.calibrated.ckpt`**. Every checkpoint listed
> below predates this protocol (focal loss, hard spectral norm, AUPRC selection) -- see
> docs/KNOWN_ISSUES.md.

baseline_acevedo:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_acevedo/runs/2026-08-25_14-31-36/checkpoints/best.ckpt
```

baseline_acevedo_ensemble:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_acevedo/ensemble_members/2026-08-25_15-31-22/checkpoints/ensemble.ckpt
```

sngp_acevedo:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/runs/2026-08-25_14-32-45/checkpoints/best.ckpt
```

sngp_acevedo_ensemble:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/ensemble_members/2026-08-25_15-34-35/checkpoints/ensemble.ckpt
```

---

baseline_wong:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_wong/runs/2026-08-27_14-23-47/checkpoints/best.ckpt
```

baseline_wong_ensemble:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_wong/ensemble_members/2026-08-27_14-37-17/checkpoints/ensemble.ckpt
```

sngp_wong:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/sngp_classifier_wong/runs/2026-08-31_16-36-39/checkpoints/best.ckpt
```

sngp_wong_ensemble:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/sngp_classifier_wong/ensemble_members/2026-08-27_14-43-32/checkpoints/ensemble.ckpt
```

---

Wong (ucdavis) models. Each run is the in-distribution test split of the UC Davis
institution subset (`data=wong data.datamodule.institution=ucdavis fold=test`); the
`wong_ucdavis` leaf comes from `data.name=wong_ucdavis`, set so per-institution runs
off the same checkpoint don't collide in a shared `wong/` folder.

baseline_wong_ucdavis:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_wong/runs/2026-09-01_16-56-30/checkpoints/best.ckpt
```

baseline_wong_ucdavis_ensemble:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_wong/ensemble_members/2026-09-01_17-00-38/checkpoints/ensemble.ckpt
```

sngp_wong_ucdavis:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/sngp_classifier_wong/runs/2026-09-01_16-58-05/checkpoints/best.ckpt
```

sngp_wong_ucdavis_ensemble:
```bash
/data1/maheswararao/experiments/uncertaity-aware-ml/train/sngp_classifier_wong/ensemble_members/2026-09-01_17-00-17/checkpoints/ensemble.ckpt
```

---

**Acevedo SNGP spectral-norm-bound ablation (2026-09-17).** A one-factor sweep of
`model.net.spectral_norm_bound` over `{0.9, 0.95, 1.0, 2.0, 4.0, 6.0, 8.0, None}`, with
everything else at `experiment=sngp_acevedo` and seed 12345 in every run. `None` is the
unbounded (`σ ≡ 1`) regime, not a bound value. Uncalibrated only -- no
`best.calibrated.ckpt` was fit for these, since the ablation is read on accuracy, which
no post-hoc knob can move. Ablation figure and table:
[results/ACEVEDO_RESULTS.md](results/ACEVEDO_RESULTS.md); inference outputs:
[MASTER_INFER_RESULTS_PATH.md](MASTER_INFER_RESULTS_PATH.md).

sngp_acevedo_snb_ablation (8 runs, one subdirectory per bound):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_classifier_acevedo/multiruns/2026-09-17_15-06-32_spectral_norm_bound_ablation/spectral_norm_bound_<c>/checkpoints/best.ckpt
```

Note that `configs/experiment/sngp_acevedo.yaml` pins `spectral_norm_bound: 4.0` from the
W&B re-sweep, but 4.0 is the *worst* bound in this grid on both Acevedo test accuracy and
the `val/nll_cal_best` selection metric -- see the ablation section in
[results/ACEVEDO_RESULTS.md](results/ACEVEDO_RESULTS.md) before reusing that pin.
