# Datasets

All datasets are HuggingFace `datasets` hosted under the `nirschl-lab` org, curated
from [this paper](https://huggingface.co/papers/2407.01791). They share one schema
(image + label), so adding a dataset to the framework is a config change, never new
Python — see [Adding a dataset](#adding-a-dataset) below.

| Dataset | HF id | Classes | Content | Data config | Experiment preset |
|---|---|---|---|---|---|
| Acevedo et al. 2020 | `nirschl-lab/acevedo_et_al_2020` | 8 | White blood cells | `configs/data/acevedo.yaml` | ✅ `configs/experiment/*_acevedo.yaml` |
| Tang et al. 2019 | `nirschl-lab/tang_et_al_2019` | 4 | Amyloid plaques | `configs/data/tang.yaml` | ✅ `configs/experiment/*_tang.yaml` |
| Wong et al. 2022 | `nirschl-lab/wong_et_al_2022` | 4 | Amyloid plaques | `configs/data/wong.yaml` | ✅ `configs/experiment/*_wong.yaml` |
| Kather et al. 2018 | `nirschl-lab/kather_et_al_2018` | 9 | Colorectal histology | `configs/data/kather2018.yaml` | ✅ `configs/experiment/*_kather2018.yaml` |
| Jung et al. 2022 | `nirschl-lab/jung_et_al_2022` | 5 | White blood cells | `configs/data/jung.yaml` | — no preset yet |
| Nirschl et al. 2018 | `nirschl-lab/nirschl_et_al_2018` | 2 | Cardiac tissue | `configs/data/nirschl2018.yaml` | — no preset yet |
| Kather et al. 2016 | `nirschl-lab/kather_et_al_2016` | 8 | Colorectal histology | `configs/data/kather2016.yaml` | — no preset yet |

Every dataset has its own `configs/data/<dataset>.yaml` — the single source of truth
for that dataset's `dataset_name`/`num_classes`/`class_to_idx`. Datasets without an
experiment preset still work: the datamodule schema is uniform, so composing the data
config with a model config directly is enough, no manual dataset_name/num_classes
overrides needed:

```bash
uv run src/train.py data=jung model=baseline_classifier
```

## In-distribution vs. out-of-distribution evaluation

Every dataset serves double duty:

- **In-distribution**: train and evaluate on the same dataset (its own train/val/test
  folds).
- **Out-of-distribution**: train on one dataset, evaluate on another. This is how the
  project's OOD/uncertainty claims are measured — see
  [DEVELOPMENT.md#metrics--visualization](DEVELOPMENT.md#metrics--visualization) for
  the cross-dataset AUROC-OOD computation this feeds.

## Adding a dataset

Add a `configs/data/<dataset>.yaml` (copy an existing one and change `dataset_name`,
`num_classes`, `class_to_idx`, and the top-level `name:` — a short, path-safe
identifier used for output-directory naming, see
[docs/OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md)), then reference it from an experiment file
via `override /data: <dataset>` — or compose ad hoc on the CLI for a dataset with no
preset yet. No new Python is needed — the datamodule
(`src/data/classification_image_datamodule.py`) is dataset-agnostic. See
[KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the two known `ClassificationImageDataModule`
caveats (duplicate class name across two files, a `setup()` edge case with
`stage=None`).

## Artifact-robustness evaluation

A separate axis, orthogonal to which dataset is used: `configs/artifact/` holds
sampling profiles from the external `histo-artifact-sim` package for simulating
imaging artifacts (stain variation, blur, etc.) on any of the above datasets, paired
with `configs/data/artifact_image_classifier.yaml` for paired real/simulated
inference. See [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)'s artifact-mode section and
`configs/artifact/README.md`.
