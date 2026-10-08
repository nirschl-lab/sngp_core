> **SNGP checkpoints below predate the canonical-SNGP correction.** They were trained
> with the old GP head (mean-field applied inside the training loss, `cov_ema` instead of
> a precision matrix) and will **not** load on the `sngp-corrections` branch or later --
> `load_state_dict` raises with a pointer to the `sngp-pre-correction` tag. Check out that
> tag, or the `isbi2026` branch, to use them; they remain the ISBI 2026 paper's
> checkpoints. Baseline and Deep-Ensemble checkpoints are unaffected.

> **Per-dataset pages.** This file stays the entry point, but detail for newer datasets
> lives under `docs/checkpoints/`. Currently:
> [checkpoints/CIFAR_CHECKPOINTS.md](checkpoints/CIFAR_CHECKPOINTS.md) — the CIFAR-100 /
> WideResNet-28-10 benchmark arms, which are off-protocol and have no
> `best.calibrated.ckpt` (they pin `mean_field_factor` instead of fitting it).

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

sngp_acevedo_snb_c6_calibrated (2026-09-29). The one exception to "uncalibrated only" is
`c* = 6.0`, the val-selected bound, with `mean_field_factor` 0.4909 fit on val NLL. It is
used in the calibrated comparison of
[results/ACEVEDO_SPECREG_RESULTS.md](results/ACEVEDO_SPECREG_RESULTS.md):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_classifier_acevedo/multiruns/2026-09-17_15-06-32_spectral_norm_bound_ablation/spectral_norm_bound_6.0/checkpoints/best.calibrated.ckpt
```

Note that `configs/experiment/sngp_acevedo.yaml` pins `spectral_norm_bound: 4.0` from the
W&B re-sweep, but 4.0 is the *worst* bound in this grid on both Acevedo test accuracy and
the `val/nll_cal_best` selection metric -- see the ablation section in
[results/ACEVEDO_RESULTS.md](results/ACEVEDO_RESULTS.md) before reusing that pin.

---

**Acevedo SNGP + spectral regularization pilot (2026-09-18).** First run of the
rep-spectral variant (`experiment=sngp_specreg_acevedo`, branch `sngp-spectral-reg`,
[models/SNGP_SPECREG_GUIDE.md](models/SNGP_SPECREG_GUIDE.md)): no spectral normalization
anywhere, `CE + 0.01 * sum sigma_max^2` over the backbone every 24 steps after a 50-epoch
burn-in, 100 epochs, AdamW lr 1e-3 with weight decay 0, no early stopping, and `best.ckpt`
= min raw `val/nll` over the regularized epochs only (`ModelCheckpointFromEpoch`). Off the
fair-comparison protocol in every one of those respects, deliberately; see the results doc.
**Uncalibrated by decision** -- no `best.calibrated.ckpt` was fit, so this entry departs
from the "report inference from `best.calibrated.ckpt`" convention above: every number in
[results/ACEVEDO_SPECREG_RESULTS.md](results/ACEVEDO_SPECREG_RESULTS.md) comes from
`best.ckpt` as saved (`mean_field_factor 1.0`). W&B run `5bnnbxdb`, group `SpectralReg`.

sngp_specreg_acevedo_v1 (epoch 88 of 0-99, val/nll 0.0732):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_acevedo/runs/2026-09-18_14-11-49/checkpoints/best.ckpt
```

sngp_specreg_acevedo_v1_last (epoch 99, the paper-style end-of-training model):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_acevedo/runs/2026-09-18_14-11-49/checkpoints/last.ckpt
```

sngp_specreg_acevedo_v1_calibrated (2026-09-29, `mean_field_factor` 0.2455 fit on val NLL). This
is used only by the calibrated comparison section of the results doc:
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_acevedo/runs/2026-09-18_14-11-49/checkpoints/best.calibrated.ckpt
```

The sibling run directories `2026-09-18_13-59-*` and `2026-09-18_14-01-*` under the same
parent are smoke tests (`fast_dev_run`, 3-epoch mini runs), not results.

---

**Acevedo GP head without SN, trained with Muon (2026-09-30).** Three arms on the
`sngp_acevedo` protocol, with no spectral normalization and no spectral penalty
(`scripts/tmux/acevedo_muon.sh`, branch `acevedo-muon`, W&B group
`Acevedo_muon_2026-09-30_12-47-49`). Each `best.calibrated.ckpt` carries `mean_field_factor` fit on
val NLL. Results: [results/ACEVEDO_MUON_RESULTS.md](results/ACEVEDO_MUON_RESULTS.md).
Run dirs `*_smoke_*` under the same parents are smoke tests, not results.

sngp_muon_wd0_acevedo (`experiment=sngp_muon_acevedo model.optimizer.weight_decay=0.0`, epoch 62, mff 7.3992):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_classifier_acevedo/runs/2026-09-30_12-47-49_muon_wd0/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_classifier_acevedo/runs/2026-09-30_12-47-49_muon_wd0/checkpoints/best.calibrated.ckpt
```

sngp_muon_wd0.1_acevedo (`experiment=sngp_muon_acevedo`, epoch 139, mff 1.4269):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_classifier_acevedo/runs/2026-09-30_12-47-49_muon_wd0.1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_classifier_acevedo/runs/2026-09-30_12-47-49_muon_wd0.1/checkpoints/best.calibrated.ckpt
```

sngp_nosn_adamw_acevedo (control: `experiment=sngp_acevedo model.net.use_spectral_norm=false model.net.spectral_norm_bound=null`, epoch 36, mff 0):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_nosn_classifier_acevedo/runs/2026-09-30_12-47-49_adamw_nosn/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_nosn_classifier_acevedo/runs/2026-09-30_12-47-49_adamw_nosn/checkpoints/best.calibrated.ckpt
```

---

**Wong ADRC study, final runs (2026-10-05).** One cell per arm from the `adrc_wong_*` grid
sweeps ([MASTER_SWEEPS.md](MASTER_SWEEPS.md)) plus a deterministic SGD baseline, each over seeds
{12345, 1, 2, 3, 4}: 150 epochs, early stopping off, SGD + Nesterov, GP arms with
`likelihood=binary_logistic` and `mean_field_factor` pinned at π/8 (no calibrated checkpoint).
Launcher `scripts/tmux/adrc_wong_final.sh`, W&B group `adrc_final_2026-10-05_10-05-27`, tag
`wong_adrc_5seed`. Results: [results/WONG_ADRC_RESULTS.md](results/WONG_ADRC_RESULTS.md).
Run dirs `*_smoke_*` under the same parents are smoke tests, not results.

adrc_baseline_wong (`experiment=baseline_wong_sgd`; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong/runs/2026-10-05_10-05-27_baseline_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong/runs/2026-10-05_10-05-27_baseline_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong/runs/2026-10-05_10-05-27_baseline_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong/runs/2026-10-05_10-05-27_baseline_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong/runs/2026-10-05_10-05-27_baseline_s4/checkpoints/best.ckpt
```

adrc_sngp_wong (`experiment=sngp_wong_sgd`, c=1, σ²=1; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong/runs/2026-10-05_10-05-27_sngp_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong/runs/2026-10-05_10-05-27_sngp_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong/runs/2026-10-05_10-05-27_sngp_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong/runs/2026-10-05_10-05-27_sngp_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong/runs/2026-10-05_10-05-27_sngp_s4/checkpoints/best.ckpt
```

adrc_bnsn_wong (`experiment=sngp_bnsn_wong_sgd`, c=8 (BN cap tied), σ²=1; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong/runs/2026-10-05_10-05-27_bnsn_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong/runs/2026-10-05_10-05-27_bnsn_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong/runs/2026-10-05_10-05-27_bnsn_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong/runs/2026-10-05_10-05-27_bnsn_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong/runs/2026-10-05_10-05-27_bnsn_s4/checkpoints/best.ckpt
```

adrc_specreg_wong (`experiment=sngp_specreg_wong_sgd`, λ=0.003, σ²=1; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong/runs/2026-10-05_10-05-27_specreg_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong/runs/2026-10-05_10-05-27_specreg_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong/runs/2026-10-05_10-05-27_specreg_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong/runs/2026-10-05_10-05-27_specreg_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong/runs/2026-10-05_10-05-27_specreg_s4/checkpoints/best.ckpt
```

adrc_muon_wong (`experiment=sngp_muon_wong_sgd`, Muon wd=0.01, σ²=1; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-05_10-05-27_muon_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-05_10-05-27_muon_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-05_10-05-27_muon_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-05_10-05-27_muon_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-05_10-05-27_muon_s4/checkpoints/best.ckpt
```

adrc_muon2stage_wong (`experiment=sngp_muon_2stage_wong_sgd`, Muon wd=0.01 → SGD + Nesterov at epoch 120, cosine, aux SGD lr 0.01, σ²=1; seeds 12345, 1, 2, 3, 4; launched 2026-10-06 with `ARMS=muon2stage`):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong/runs/2026-10-06_21-21-42_muon2stage_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong/runs/2026-10-06_21-21-42_muon2stage_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong/runs/2026-10-06_21-21-42_muon2stage_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong/runs/2026-10-06_21-21-42_muon2stage_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong/runs/2026-10-06_21-21-42_muon2stage_s4/checkpoints/best.ckpt
```

---

**Wong ADRC study, UC Davis-only reruns (2026-10-05).** The same 5 arms, cells, seeds and recipe as
the full-Wong final runs above, trained on `data.datamodule.institution=ucdavis` only (cells not
re-tuned on UC Davis). Launcher `INSTITUTION=ucdavis NUM_WORKERS=4 scripts/tmux/adrc_wong_final.sh`,
W&B group `adrc_final_ucdavis_2026-10-05_21-33-51`, tag `wong_ucdavis_adrc_5seed`. Results:
"UC Davis-trained runs" in [results/WONG_ADRC_RESULTS.md](results/WONG_ADRC_RESULTS.md). The
`2026-10-05_21-29-27_smoke_*` run dirs under the same parents are smoke tests, not results.

adrc_baseline_wong_ucdavis (`experiment=baseline_wong_sgd`; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_baseline_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_baseline_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_baseline_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_baseline_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_baseline_s4/checkpoints/best.ckpt
```

adrc_sngp_wong_ucdavis (`experiment=sngp_wong_sgd`, c=1, σ²=1; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_sngp_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_sngp_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_sngp_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_sngp_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_sngp_s4/checkpoints/best.ckpt
```

adrc_bnsn_wong_ucdavis (`experiment=sngp_bnsn_wong_sgd`, c=8 (BN cap tied), σ²=1; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_bnsn_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_bnsn_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_bnsn_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_bnsn_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_bnsn_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_bnsn_s4/checkpoints/best.ckpt
```

adrc_specreg_wong_ucdavis (`experiment=sngp_specreg_wong_sgd`, λ=0.003, σ²=1; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_specreg_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_specreg_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_specreg_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_specreg_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_specreg_s4/checkpoints/best.ckpt
```

adrc_muon_wong_ucdavis (`experiment=sngp_muon_wong_sgd`, Muon wd=0.01, σ²=1; seeds 12345, 1, 2, 3, 4):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_muon_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_muon_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_muon_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_muon_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong_ucdavis/runs/2026-10-05_21-33-51_muon_s4/checkpoints/best.ckpt
```

adrc_muon2stage_wong_ucdavis_5seed (as adrc_muon2stage_wong, trained on UC Davis only; seeds 12345, 1, 2, 3, 4; launched 2026-10-06 with `INSTITUTION=ucdavis ARMS=muon2stage`. Its s12345 is a retrain, separate from the seed-12345-only adrc_muon2stage_wong_ucdavis run):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_21-21-44_muon2stage_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_21-21-44_muon2stage_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_21-21-44_muon2stage_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_21-21-44_muon2stage_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_21-21-44_muon2stage_s4/checkpoints/best.ckpt
```

adrc_muon2stage_wong_ucdavis (`experiment=sngp_muon_2stage_wong_sgd`, Muon wd=0.01 → SGD + Nesterov at epoch 120, σ²=1; seed 12345 only):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_11-52-14_muon2stage_s12345/checkpoints/best.ckpt
```

adrc_muon2stage_wsd_wong_ucdavis (`experiment=sngp_muon_2stage_wsd_wong_sgd`, as adrc_muon2stage_wong_ucdavis but a WSD inverse-proportional decay to 0.1× from epoch 120; seed 12345 only):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_15-30-19_muon2stage_wsd_s12345/checkpoints/best.ckpt
```

adrc_muon2stage_wsd_aux04_wong_ucdavis (`experiment=sngp_muon_2stage_wsd_wong_sgd model.optimizer.sgd_lr=0.04`, as adrc_muon2stage_wsd_wong_ucdavis with the aux SGD lr at 0.04; seed 12345 only):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_16-25-59_muon2stage_wsd_aux04_s12345/checkpoints/best.ckpt
```

adrc_muon2stage_wsd_aux02_wong_ucdavis (as adrc_muon2stage_wsd_aux04_wong_ucdavis with `model.optimizer.sgd_lr=0.02`; seed 12345 only):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-06_20-17-27_muon2stage_wsd_aux02_s12345/checkpoints/best.ckpt
```

adrc_muon2stage_sw135_wong_ucdavis (as adrc_muon2stage_wong_ucdavis with `model.scheduler.cosine_start_epoch=135`: Muon → SGD switch + cosine 0.01 → 1e-4 at epoch 135; seed 12345 only; arm `muon2stage_sw135`):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong_ucdavis/runs/2026-10-07_11-13-59_muon2stage_sw135_s12345/checkpoints/best.ckpt
```

adrc_muon2stage_sw135_wong (as adrc_muon2stage_wong with `model.scheduler.cosine_start_epoch=135`: Muon → SGD switch + cosine 0.01 → 1e-4 at epoch 135; full Wong, seed 12345 only; arm `muon2stage_sw135`):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_2stage_classifier_wong/runs/2026-10-07_11-13-57_muon2stage_sw135_s12345/checkpoints/best.ckpt
```

adrc_muon_aux01_wong (`experiment=sngp_muon_wong_sgd model.optimizer.weight_decay=0.01 model.optimizer.sgd_lr=0.01`: Muon wd=0.01 on the hidden convs, SGD + Nesterov lr 0.01 on stem / BN / GP output layer, σ²=1; seeds 12345, 1, 2, 3, 4; arm `muon_aux01`):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-07_15-26-25_muon_aux01_s12345/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-07_15-26-25_muon_aux01_s1/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-07_15-26-25_muon_aux01_s2/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-07_15-26-25_muon_aux01_s3/checkpoints/best.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong/runs/2026-10-07_15-26-25_muon_aux01_s4/checkpoints/best.ckpt
```

adrc_muon_aux01_wong_ucdavis (as adrc_muon_aux01_wong, trained on UC Davis only; seed 12345 only; W&B `78kdh5jb`; best.ckpt = epoch 82, so last.ckpt is reported too):
```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_muon_sgd_classifier_wong_ucdavis/runs/2026-10-07_13-05-26_muon_aux01_s12345/checkpoints/best.ckpt
```
