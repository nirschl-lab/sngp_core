# CIFAR-100 benchmark checkpoints

Checkpoints for the CIFAR-100 / WideResNet-28-10 reproduction of the SNGP benchmark.
What the runs are and how to reproduce them:
[../models/CIFAR100_BENCHMARK.md](../models/CIFAR100_BENCHMARK.md). Entry point for every
other dataset's checkpoints: [../MASTER_CHECKPONT_PATHS.md](../MASTER_CHECKPONT_PATHS.md).

> **Convention for this page, which differs from the master doc.** These runs have
> **no `best.calibrated.ckpt`** — the CIFAR arms skip the post-hoc calibration pass and
> pin `mean_field_factor` at the reference's CIFAR-100 value of 7.5 instead. Record two
> checkpoints per run and report both:
>
> - **`best.ckpt`** — lowest `val/loss` (for the spectral-reg arms, ranked only over
>   epochs ≥ the burn-in, via `ModelCheckpointFromEpoch`). The primary.
> - **`last.ckpt`** — epoch 249, which is what the reference implementation reports.
>
> Everything on this page is **off the biomedical protocol** (SGD + piecewise schedule,
> `val/loss` selection, no early stopping, reference GP constants). Do not report these
> numbers next to the protocol runs without saying so.

Run directories follow the standard layout
([../OUTPUT_LAYOUT.md](../OUTPUT_LAYOUT.md)):

```
${EXPERIMENTS_HOME}/${PROJECT_NAME}/train/<model.name>_cifar100/runs/<timestamp>/checkpoints/
```

so the four arms land under `train/baseline_classifier_cifar100`,
`train/sngp_classifier_cifar100` and `train/sngp_specreg_classifier_cifar100` (both
spectral-reg arms share that last one — tell them apart by timestamp and by the W&B run
name, `cifar100_sngp_specreg_wrn28x10` vs `cifar100_sngp_specreg_literal_wrn28x10`).

Verify any entry before using it:

```bash
uv run python -c "from src.checkpointing.io import read_meta; print(read_meta('<path>'))"
# net_spec.arch must be 'wide_resnet28_10'; num_classes 100
```

---

All four arms were trained 2026-09-20 in parallel, one per L40S, 250 epochs, seed 12345,
W&B group `CIFAR100`. Results: [../results/CIFAR100_RESULTS.md](../results/CIFAR100_RESULTS.md).

> **Run-directory collision — read before using the spectral-reg checkpoints.** Both
> spectral-reg arms had `model.name: sngp_specreg_classifier`, so `task_name` was
> identical, and launching them in the same second gave them the *same* Hydra run
> directory. Two trainers wrote into one `checkpoints/`: Lightning suffixed the second
> `best` as `best-v1.ckpt`, and **the literal arm's `last.ckpt` was overwritten** by the
> matched arm's and is gone. The files below are identified by their `hyper_parameters`
> (`spec_reg_burnin_epochs`), not by filename order. Fixed for future runs by
> `model.name: sngp_specreg_literal_classifier` in
> `configs/experiment/sngp_specreg_cifar100_literal.yaml`; re-running the literal arm
> would give it a clean directory and recover its `last.ckpt`.

## baseline_cifar100 — deterministic WRN-28-10

```bash
# best.ckpt  (epoch 75, min val/loss)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/best.ckpt
# last.ckpt  (epoch 249 -- the reference reporting point)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/baseline_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt
```

## sngp_cifar100 — SNGP, reference recipe (`spectral_norm_bound` 6.0)

```bash
# best.ckpt  (epoch 76, min val/loss)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/best.ckpt
# last.ckpt  (epoch 249)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt
```

## sngp_specreg_cifar100 — spectral regularization, matched to the SNGP arm

`spec_reg_coef` 0.01, burn-in 1, weight decay 6e-4.

```bash
# best.ckpt  (epoch 246, min val/loss over epochs >= burn-in)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/best.ckpt
# last.ckpt  (epoch 249)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/last.ckpt
```

## sngp_specreg_cifar100_literal — rep-spectral paper-literal (burn-in 200, wd 0)

Shares the directory above; **its checkpoint is the `-v1` one** (`spec_reg_burnin_epochs:
200`). Its `last.ckpt` does not exist — see the collision note.

```bash
# best.ckpt  (epoch 200, the first rankable epoch)
/data1/maheswararao/experiments/uncertainty-aware-ml/train/sngp_specreg_classifier_cifar100/runs/2026-09-20_18-06-59/checkpoints/best-v1.ckpt
```

## rf_e2e — SpecReg per random-feature head, seed 12345 (2026-09-26)

`scripts/tmux/cifar100_rf_e2e.sh`. `l2paper_*` = `sngp_specreg_cifar100_rf` (ℓ 2, σ² 7.5,
λ π/8; an off-protocol head despite the label, see its config header); `l20recipe_*` = `sngp_specreg_cifar100` + `feature_map` / `random_feature_type`
overrides. Compare at `last.ckpt` (epoch 249). `best.ckpt` epochs: l2paper cos orf/simrf
240 / 220, l20recipe positive 75 / 76, hyperbolic 76 / 75. Results:
[../results/CIFAR100_RF_E2E_RESULTS.md](../results/CIFAR100_RF_E2E_RESULTS.md).

```bash
/data1/maheswararao/experiments/uncertainty-aware-ml/train/rf_e2e/2026-09-26_16-55-28/l2paper_cos_orf/checkpoints/last.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/rf_e2e/2026-09-26_16-55-28/l2paper_cos_simrf/checkpoints/last.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/rf_e2e/2026-09-26_16-55-28/l20recipe_positive_orf/checkpoints/last.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/rf_e2e/2026-09-26_16-55-28/l20recipe_positive_simrf/checkpoints/last.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/rf_e2e/2026-09-26_16-55-28/l20recipe_hyperbolic_orf/checkpoints/last.ckpt
/data1/maheswararao/experiments/uncertainty-aware-ml/train/rf_e2e/2026-09-26_16-55-28/l20recipe_hyperbolic_simrf/checkpoints/last.ckpt
# best.ckpt alongside each
```

Verify any entry before use:

```bash
uv run python -c "import torch; ck=torch.load('<path>', map_location='cpu', weights_only=False); \
print(ck['epoch'], ck['hyper_parameters'].get('spec_reg_burnin_epochs'))"
```

---

## Overnight follow-up (2026-09-21) — seeds, the `c = 4.1` control, literal re-run

Driven by `scripts/tmux/cifar100_overnight.sh`; W&B group
`CIFAR100_overnight_2026-09-20_21-38-42`. Results pooled with the seed-12345 runs above in
[../results/CIFAR100_RESULTS.md](../results/CIFAR100_RESULTS.md).

Every run has an **explicit** `hydra.run.dir`, so no two share a directory — that is what
prevents the collision documented above from recurring.

```bash
B=/data1/maheswararao/experiments/uncertainty-aware-ml/overnight/overnight_2026-09-20_21-38-42
$B/s1_baseline/checkpoints/{best,last}.ckpt        # baseline, seed 1
$B/s2_baseline/checkpoints/{best,last}.ckpt        # baseline, seed 2
$B/s1_sngp/checkpoints/{best,last}.ckpt            # SNGP c=6.0, seed 1
$B/s2_sngp/checkpoints/{best,last}.ckpt            # SNGP c=6.0, seed 2
$B/s1_specreg/checkpoints/{best,last}.ckpt         # SpecReg matched, seed 1
$B/s2_specreg/checkpoints/{best,last}.ckpt         # SpecReg matched, seed 2
$B/c41_sngp/checkpoints/{best,last}.ckpt           # SNGP spectral_norm_bound=4.1, seed 12345
$B/literal_rerun/checkpoints/{best,last}.ckpt      # rep-spectral literal, seed 12345
```

`literal_rerun` is the clean re-run of the arm whose `last.ckpt` was lost to the
collision; it has both checkpoints.

---

## `trace_logistic` likelihood arm (2026-09-23) — SpecReg matched, three seeds

Driven by `scripts/tmux/cifar100_trace_logistic.sh`; W&B group
`CIFAR100_trace_logistic_2026-09-23_17-06-47`. `sngp_specreg_cifar100` with
`model.net.likelihood=trace_logistic` — the GP head's Laplace weight becomes
`1 - ||p||²` instead of the reference's unit weight
([../models/SNGP_GUIDE.md](../models/SNGP_GUIDE.md#the-laplace-weight-likelihood)).

```bash
B=/data1/maheswararao/experiments/uncertainty-aware-ml/overnight/trace_logistic_2026-09-23_17-06-47
$B/tl_specreg_s12345/checkpoints/{best,last}.ckpt   # SpecReg matched, trace_logistic, seed 12345
$B/tl_specreg_s1/checkpoints/{best,last}.ckpt       # ... seed 1
$B/tl_specreg_s2/checkpoints/{best,last}.ckpt       # ... seed 2
```

Verify the mode survived the round trip before using one:

```bash
uv run python -c "from src.checkpointing.io import read_meta; print(read_meta('<path>').net_spec['likelihood'])"
# -> trace_logistic
```

Two caveats specific to this arm, both from `likelihood` being an inference-only knob:

- **`best.ckpt` is not comparable to the gaussian arms' `best.ckpt`.** `val/loss` is
  computed from mean-field logits, so the selected epoch differs even though the
  per-epoch weights do not. Use `last.ckpt` (epoch 249), as the rest of this page already
  recommends.
- **Do not report these at the pinned `mean_field_factor = 7.5`.** A logistic weight
  shrinks the precision accumulator ~50× on a converged CIFAR-100 classifier, so the
  predictive variance — and the correction — inflate by about as much. Start from
  `scripts/metrics/cifar100_mean_field_sweep.py` (free, offline).

## evidence_ls — SpecReg matched at the evidence-picked ℓ = 7, three seeds (2026-09-28)

`scripts/tmux/cifar100_evidence_ls.sh`: `sngp_specreg_cifar100` + `model.net.length_scale=7.0`,
nothing else changed (σ² 1, unscaled features, ridge 1.0, `gaussian`). ℓ = 7 is the type-II
evidence pick at rff_dim 1024
([../results/CIFAR100_LENGTH_SCALE_EVIDENCE_RESULTS.md](../results/CIFAR100_LENGTH_SCALE_EVIDENCE_RESULTS.md)).
W&B group `CIFAR100_evidence_ls_2026-09-28_16-10-10`, runs `ep99xl9h` / `3vul5vde` / `c7loebia`.
Compare at `last.ckpt` (epoch 249). Results:
[../results/CIFAR100_EVIDENCE_LS_RESULTS.md](../results/CIFAR100_EVIDENCE_LS_RESULTS.md).

```bash
B=/data1/maheswararao/experiments/uncertainty-aware-ml/overnight/evidence_ls_2026-09-28_16-10-10
$B/els_specreg_s12345/checkpoints/{best,last}.ckpt   # seed 12345
$B/els_specreg_s1/checkpoints/{best,last}.ckpt       # seed 1
$B/els_specreg_s2/checkpoints/{best,last}.ckpt       # seed 2
```

Verify: `read_meta('<path>').net_spec['length_scale']` → `7.0`.

## online_ls — SpecReg matched, ℓ tuned online by type-II evidence, three seeds (2026-09-29)

`scripts/tmux/cifar100_online_ls.sh` (branch `sngp-online-length-scale`): `sngp_specreg_cifar100`
starting at ℓ = 20, plus the `OnlineLengthScaleEvidence` callback (updates at epochs 10–155, fixed
from 160). Nothing else changed. The final ℓ is in each checkpoint's net spec.
W&B group `CIFAR100_online_ls_2026-09-29_12-07-14`, runs `6najd0ku` / `akc2e5l1` / `0twvirza`.
Compare at `last.ckpt` (epoch 249). Results:
[../results/CIFAR100_ONLINE_LS_RESULTS.md](../results/CIFAR100_ONLINE_LS_RESULTS.md).

```bash
B=/data1/maheswararao/experiments/uncertainty-aware-ml/overnight/online_ls_2026-09-29_12-07-14
$B/ols_specreg_s12345/checkpoints/{best,last}.ckpt   # seed 12345, final l 5.61
$B/ols_specreg_s1/checkpoints/{best,last}.ckpt       # seed 1,     final l 5.61
$B/ols_specreg_s2/checkpoints/{best,last}.ckpt       # seed 2,     final l 5.00
```

Verify: `read_meta('<path>').net_spec['length_scale']` → `5.612…` / `5.612…` / `5.0`.

## muon — GP head on an unconstrained WRN-28-10, trained with Muon, ℓ = 7, seed 12345 (2026-09-30)

`scripts/tmux/cifar100_muon.sh` (branch `acevedo-muon`): no spectral normalization and no
spectral penalty; Muon (lr 0.02) on the hidden convs, AdamW on the rest, ℓ = 7. Two schedules:
`sngp_muon_cifar100` (piecewise, tag `muon_2026-09-30_15-20-01`, W&B group
`CIFAR100_muon_2026-09-30_15-20-01`, runs `bousli1o` wd 0 / `sjbaklag` wd 0.1) and
`sngp_muon_cifar100_wsd` (`--wsd`, tag `muon_wsd_2026-09-30_15-54-17`, W&B group
`CIFAR100_muon_wsd_2026-09-30_15-54-17`, runs `5ljnhcdw` wd 0 / `9yk8zf4a` wd 0.1).
Compare at `last.ckpt` (epoch 249). Results:
[../results/CIFAR100_MUON_RESULTS.md](../results/CIFAR100_MUON_RESULTS.md).

```bash
P=/data1/maheswararao/experiments/uncertainty-aware-ml/overnight/muon_2026-09-30_15-20-01
W=/data1/maheswararao/experiments/uncertainty-aware-ml/overnight/muon_wsd_2026-09-30_15-54-17
$P/muon_wd0_s12345/checkpoints/{best,last}.ckpt     # piecewise, Muon wd 0
$P/muon_wd0.1_s12345/checkpoints/{best,last}.ckpt   # piecewise, Muon wd 0.1
$W/muon_wd0_s12345/checkpoints/{best,last}.ckpt     # WSD, Muon wd 0
$W/muon_wd0.1_s12345/checkpoints/{best,last}.ckpt   # WSD, Muon wd 0.1
```

The `muon_smoke_*` / `muon_wsd_smoke_*` trees under `overnight/` are 1200-step smoke tests, not results.

## cosine — SNGP, SNGP + SpecReg and the Muon GP head on the cosine schedule, ℓ = 7, seed 12345 (2026-10-01)

`scripts/tmux/cifar100_cosine.sh` (branch `acevedo-muon`, commit `dcf15a3`): all three arms on
the Acevedo schedule (`CosineAnnealingLR` to 0 over 250 epochs, per epoch, no warmup) and at
ℓ = 7. Experiments `sngp_cifar100_cosine` (SN c = 6.0, SGD), `sngp_specreg_cifar100_cosine`
(spectral penalty, SGD) and `sngp_muon_cifar100_cosine` (no SN, Muon wd 0.1). Tag
`cosine_2026-10-01_10-17-06`, W&B group `CIFAR100_cosine_2026-10-01_10-17-06`, runs `l602qimv`
SNGP / `1damijv6` SpecReg / `7qd4u08k` Muon wd 0.1 / `2bsnyc36` Muon wd 0. The wd-0 arm was added
an hour later (`ARMS=cos_muon_wd0_s12345` with the same stamp, driver log `driver_add_11-29-00.log`);
its AdamW group keeps weight decay 0.01, as in the piecewise `muon_wd0` arm. **It was stopped at
epoch 152**, so its `last.ckpt` is partial and not a result: the AdamW aux group let the final BN γ and
the GP output layer drift (see `configs/experiment/sngp_muon_sgd_cifar100_cosine.yaml`). It was
replaced by two `MuonWithAuxSGD` arms (experiment `sngp_muon_sgd_cifar100_cosine`, commit `7db53ea`):
the SGD arms' exact rule (0.04, Nesterov, L2 6e-4) on the stem / BN / GP output layer. They are
`cos_muonsgd_wd0.1_s12345` (W&B `8leo4avq`) and `cos_muonsgd_wd0_s12345` (W&B `jjwe330m`), launched
13:13 into the same tag (driver log `driver_add_13-13-11.log`).

Two BatchNorm-control arms on the Muon wd 0.1 + SGD aux recipe were added at 17:03 (commit `e6074b3`,
driver log `driver_add_17-03-18.log`):
- `cos_muonsgd_bnsn_wd0.1_s12345` (W&B `2debw4em`), experiment `sngp_muon_sgd_bnsn_cifar100_cosine`:
  every BN is a `SpectralBatchNorm2d` with its gain max|γ|/√(running_var+ε) capped at 3 (DUE).
- `cos_muonsgd_bnreg_wd0.1_s12345` (W&B `nkkg01xu`), experiment `sngp_muon_sgd_bnreg_cifar100_cosine`:
  the loss gets CE + 0.01·Σ_l (BN gain)² every 24 steps (`spec_reg_target: batchnorm`).

Compare at `last.ckpt` (epoch 249).

```bash
C=/data1/maheswararao/experiments/uncertainty-aware-ml/overnight/cosine_2026-10-01_10-17-06
$C/cos_sngp_l7_s12345/checkpoints/{best,last}.ckpt      # SNGP, c = 6.0
$C/cos_specreg_l7_s12345/checkpoints/{best,last}.ckpt   # SNGP + SpecReg
$C/cos_muon_wd0.1_s12345/checkpoints/{best,last}.ckpt   # GP head, no SN, Muon wd 0.1
$C/cos_muon_wd0_s12345/checkpoints/{best,last}.ckpt     # GP head, no SN, Muon wd 0 -- STOPPED at epoch 152
$C/cos_muonsgd_wd0.1_s12345/checkpoints/{best,last}.ckpt  # GP head, no SN, Muon wd 0.1 + SGD aux
$C/cos_muonsgd_wd0_s12345/checkpoints/{best,last}.ckpt    # GP head, no SN, Muon wd 0 + SGD aux
$C/cos_muonsgd_bnsn_wd0.1_s12345/checkpoints/{best,last}.ckpt   # Muon wd 0.1 + SGD aux + BN spectral norm (c = 3)
$C/cos_muonsgd_bnreg_wd0.1_s12345/checkpoints/{best,last}.ckpt  # Muon wd 0.1 + SGD aux + BN SpecReg (0.01, every 24 steps)
```

The `cosine_smoke_*` tree under `overnight/` is a 1200-step smoke test, not a result.
