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

## Filtering by institution

Some datasets (e.g. Wong et al. 2022, Tang et al. 2019) mix rows sourced from
multiple institutions in a per-row `institution` HF column. Pass
`data.datamodule.institution=<id>` to scope training/eval/inference to a single
institution's rows; leave it unset (the default, `null`) to train on all data as
before:

```bash
uv run src/train.py experiment=baseline_wong data.datamodule.institution=ucdavis
```

Requesting an institution id that doesn't match any rows, or a dataset with no
`institution` column at all, fails loudly at `setup()` time rather than silently
training on an empty split. `uv run src/visualization/datasets_class_distribution.py
--dataset <name>` reports which institution ids (if any) a given dataset actually has.

## Artifact-robustness evaluation

A separate axis, orthogonal to which dataset is used: the external `histo-artifact-sim`
package blends real artifact cutouts (debris, ink, bone, fibres, bubbles, scratches) and
procedural acquisition effects into clean patches. `configs/data/artifact_image_classifier.yaml`
wires it up for paired real/simulated inference — each test sample yields both the real
image and a simulated one, which is what
[INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)'s artifact-mode section consumes.

Sampling profiles ship **inside the package** and are selected by name:
`artifact_config_path: balanced` gives all 11 artifact categories equal occurrence (use
this for training and evaluation — under `empirical`, `focus` would appear about six
times per thousand patches, too few to measure); `empirical` reproduces real slide
prevalence, for a held-out test set meant to mirror it. A path to a policy file generated
by `histo-artifacts profile` also works.

Three inputs are machine-specific rather than config:

| | |
| --- | --- |
| `HISTO_ARTIFACTS_BANK` | the asset bank (`alpha/` + `paired/`), exposed as `${paths.artifact_bank_dir}` |
| `HISTO_ARTIFACTS_CACHE` | where the measured asset index is cached; read by the package itself |
| `data/artifact/artifact_taxonomy_by_image.csv` | per-asset categories, regenerated by `scripts/artifact/build_taxonomy.py` |

The taxonomy is not optional. The bank's folder layout puts 107 of 611 assets (`marker`,
`hair`) in categories no policy mentions, and leaves `stain_artifact` and
`processing_material` — which every policy weights — with no assets at all; the datamodule
runs with `on_missing_category="error"`, so that mismatch fails the run rather than
quietly reshaping the distribution.

**Coverage depends on the field of view.** A coverage percentage is a fraction of the
patch the artifact lands on, so the same debris fragment covers sixteen times more of a
256px patch than of a 1024px one at the same magnification. The simulator sees the
*native* image, before the datamodule's resize/crop, so `artifact_reference_patch_px`
must match that native size (360 for Acevedo) and `artifact_patch_magnification` the
optics it is read as (40x). Together those make the coverage multiplier exactly 1.0, so
the quartiles derived from the bank apply literally. Point the datamodule at a dataset
with a different native patch size and `artifact_reference_patch_px` has to move with it.
