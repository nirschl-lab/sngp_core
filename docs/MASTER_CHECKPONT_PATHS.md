<!-- generated from configs/runs/infer_manifest.yaml -- do not edit by hand.
     Regenerate with `uv run src/metrics/manifest.py --write-master-docs`. -->

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
