# Datasets

The project's own datasets are HuggingFace `datasets` hosted under the `nirschl-lab`
org, curated from [this paper](https://huggingface.co/papers/2407.01791). They share one
schema (image + label), so adding one of *those* is a config change, never new Python —
see [Adding a dataset](#adding-a-dataset) below. Three public benchmarks are also wired
up, through an adapter; see [Public benchmarks](#public-benchmarks).

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

## Public benchmarks

Added for the CIFAR-100 / WideResNet-28-10 reproduction of the SNGP benchmark — see
[models/CIFAR100_BENCHMARK.md](models/CIFAR100_BENCHMARK.md).

| Dataset | HF id | Classes | Role | Data config |
|---|---|---|---|---|
| CIFAR-100 | `uoft-cs/cifar100` | 100 | Train + in-distribution test | `configs/data/cifar100.yaml` |
| CIFAR-10 | `uoft-cs/cifar10` | 10 | OOD set only | `configs/data/cifar10.yaml` |
| SVHN | `ufldl-stanford/svhn` (`cropped_digits`) | 10 | OOD set only | `configs/data/svhn.yaml` |

**These are not config-only.** None of them is in the `nirschl-lab` schema: CIFAR-100 has
`img`/`fine_label`/`coarse_label`, none of the three has an `image_id` or a
`classes_to_idx` column, and none ships a `validation` split. They therefore load through
`src/data/benchmark_image_datamodule.py::BenchmarkImageDataModule`, which renames the
columns, synthesizes `image_id`, carves a stratified validation split out of `train`
(5000 rows, seed 42 — never out of `test`, or checkpoint selection would leak into the
reported numbers), and validates `class_to_idx` against the split's `ClassLabel` feature
instead of the missing column.

They also use `configs/img_augmentations/cifar32.yaml`, which keeps images at their native
32×32 rather than resizing to 224 like every other preset. CIFAR-10 and SVHN reuse the
CIFAR-100 preset deliberately: an OOD input must be preprocessed exactly the way the
in-distribution data was, or the measured OOD score partly reflects a preprocessing shift.

Adding another public benchmark (CIFAR-10 as a *training* set, Fashion-MNIST, …) is a
config change against `BenchmarkImageDataModule` — set `image_column`, `label_column`,
`drop_columns` and `dataset_config_name` to match its schema.

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

Setting `institution` also renames the train/eval/infer output path for that run --
`wong` + `institution=upitt` writes to `wong_upitt` instead of the shared `wong/` root
every other institution (and the unfiltered, all-institution run) would otherwise also
write to. This is automatic; no separate `data.name=` override is needed. See
[docs/OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md#2-how-task_name-and-run-id-are-generated) for
the `dataset_label` resolver that does this.

## Artifact-robustness evaluation

A separate axis, orthogonal to which dataset is used: the external `histo-artifact-sim`
package (imported as `histo_artifacts`) blends real artifact cutouts (debris, ink, bone,
fibres, bubbles, scratches) and procedural acquisition effects into clean patches.
`configs/data/artifact_image_classifier.yaml` wires it up for paired real/simulated
inference — each test sample yields both the real image and a simulated one, which is
what [INFERENCE_GUIDE.md](INFERENCE_GUIDE.md)'s artifact-mode section consumes.

### Two independent axes

As of the `d0b2ae7` package rev, simulation is two composable axes rather than one blended
policy, each with its own knob and its own way to switch it off:

| axis | what it does | policies | knob | off |
|---|---|---|---|---|
| **artifact** (`artifact_config_path`) | which real cutouts get pasted, how large, how often | `artifact_balanced` (equal per-category occurrence — use for training/eval, since under `artifact_empirical` `focus` appears only ~6 times per thousand patches, too few to measure), `artifact_empirical` (real slide prevalence — for a held-out test set meant to mirror it), or a path generated by `histo-artifacts profile` | `artifact_count` — pins an exact overlay count, overriding the policy's own clean/overlay-probability draw entirely | `artifact_config_path: none` |
| **procedural** (`artifact_procedural_config`) | which acquisition degradations (stain shift, blur, noise, JPEG compression, vignette, chromatic aberration, geometric warp, ...) fire, computed rather than pasted | `procedural_minimal` (default when left `null` — only `illumination_gradient`+`local_blur`), `procedural_standard` (+ `pixelate`, `brightness_contrast`), `procedural_ood` (all 13 effects — pair this with `artifact_severity` for graded OOD evaluation) | `artifact_severity` (1-5, via `artifact_ladder`, default the shipped `histo_c`) — grades *magnitude* of a firing effect only, independent of how often it fires | `artifact_procedural_config: none` |

The two are genuinely independent: `artifact_config_path: none` + `artifact_procedural_config: none`
is the clean baseline arm (no simulation at all — the "real" and "artifact" streams come out
identical), and every other combination of `{artifact_balanced/artifact_empirical/none} ×
{procedural_minimal/standard/ood/none}` is a valid, distinctly-attributable arm. This is what
makes "does the accuracy drop come from artifacts or from acquisition degradations?" answerable
rather than confounded into one "simulation" effect.

`predictions.csv` in artifact mode carries `count`/`severity` (the two knobs above, constant
for the whole run), `percent_pixels_affected` (per-sample, the fraction of pixels with any
nonzero alpha in the *localized* `artifact_mask` — `artifact_mask` is per-pixel blend alpha
(0-255), not a 0/1 flag, so this thresholds at `>0` rather than averaging raw alpha, which
would report blend *strength* rather than coverage. Usually 0 on a procedural-only run, since
most procedural effects are whole-frame and don't touch that mask — except `local_blur`, the
one procedural effect registered with local rather than global/warp scope, which localizes into
the mask same as a pasted overlay would), and `global_degradations`/`geometric_degradations`
(JSON-encoded lists of effect names, `[]` when none fired). **A 0 in `percent_pixels_affected`
does not mean the image was untouched** — a whole-frame effect like `hed_stain_shift` changes
every pixel's value but is never reflected in `artifact_mask` (a whole-frame percentage would
always read 0% or 100%, carrying no coverage information), so it shows up only in
`global_degradations` instead; check that column, not just `percent_pixels_affected`, to tell
"nothing simulated happened to this sample" apart from "a global effect touched it." See
"Saving sweep results" below and `.claude/skills/metrics/references/csv_schema.md` for the
full column reference.

`percent_pixels_affected` can also be 0 on the *artifact* axis, even with `artifact_count`
pinned — pinning guarantees an overlay is **attempted** (`count=N` skips the clean-probability
roll entirely, see `count` above), not that it renders visibly. A pull from a thin bank category
(the README's own `focus`/`scratch_glass`/`bubble`/`tissue_component`/`processing_material`
caveat) can composite to zero visible alpha despite a nonzero target coverage — check the
sample's `operations` metadata (`kind="overlay"`, `clipped=True`) to distinguish "no overlay
fired" from "an overlay fired but rendered invisibly." At `artifact_balanced`/`count=1` this
happens on roughly 6% of Acevedo test samples, dominated by `focus/out_of_focus` and
`stain_artifact/gms_background`.

### Saving sweep results

The real (clean) stream's predictions are bit-identical for every artifact/procedural
combination run against the same checkpoint — the simulator never touches `real_image`. Sweeping
`artifact_count`/`artifact_severity` while resaving both streams every time is therefore pure
duplication. The convention:

1. Run once per checkpoint with `infer.save.streams=[real]` (config and procedural both
   `none`) to a `real_baseline/` subfolder — this is the shared clean-prediction reference.
2. Run each swept variant with `infer.save.streams=[artifact]` — the model never even runs a
   forward pass on the real stream for these, not just a CSV-writing skip.
3. Land both under the same `<ckpt>/acevedo_artifact/` root, e.g.
   `.../acevedo_artifact/real_baseline/predictions.csv`,
   `.../acevedo_artifact/config/count_1/predictions.csv`,
   `.../acevedo_artifact/procedural/severity_1/predictions.csv` — set via
   `infer.save.run_name=...` (see [docs/OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md) §5).

`src.metrics.artifact_quantification.quantify_artifact_impact`'s optional
`real_csv_map` param joins an artifact-only sweep CSV against its checkpoint's `real_baseline`
CSV before computing paired metrics (accuracy drop, confidence drop, prediction-flip rate, ...).
A pre-sweep, self-contained dual-stream CSV (the `infer.save.streams` default) needs no entry
there.

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
