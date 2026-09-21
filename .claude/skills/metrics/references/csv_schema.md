# Prediction CSV schemas

Two different pipelines write prediction CSVs with **different column names** for the
same underlying quantities. Check which one you have before calling a metrics
function — passing the wrong schema fails with a `KeyError`, not a helpful message.

## Training-time test CSVs (`TestArtifactsCallback`)

Written by `src/callbacks/test_artifacts_callback.py` during `trainer.test(...)` when
`model.log_csv=true`, to `<csv_save_path>/<dataset_name>.csv`:

| column | meaning |
|---|---|
| `image_id` | sample identifier |
| `target` | ground-truth class index |
| `prediction` | predicted class index (argmax) |
| `prediction_prob_score` | max softmax probability (confidence) |
| `true_bin_label` | 1 if `prediction == target` else 0 |
| `class_logits` | list of raw logits per class |
| `class_probs` | list of softmax probabilities per class |
| `fold` | `train`/`validation`/`test` |
| `predictive_entropy` | float, nats. Always written |
| `confidence_margin` | float, top-1 minus top-2 probability. Always written |
| `dempster_shafer` | float in `[0, 1]`. Always written |
| `uncertainty` | float. Only when the model exposes one — same semantics and per-family units as the inference schema below |
| `uncertainty_kind` | string naming that unit. Written alongside `uncertainty` |
| `total_entropy`, `aleatoric_entropy`, `mutual_information` | floats, nats. Only when a member stack exists (Deep Ensemble, or MC-Dropout via `model.use_mc=true`) |
| `raw_logits` | list. SNGP only |

The two list columns here are written as Python list objects and stringified by pandas
(`repr`, not JSON) — which is why `_parse_class_probs` needs its `ast.literal_eval`
fallback. The inference schema below uses real `json.dumps` instead.

**Consumed by**: `src/metrics/auc.py`'s `AUROC_across_dataset` /
`AUROC_across_dataset_full_population` (read `prediction_prob_score` for
`score_mode="msp"`, `class_probs` for `score_mode="entropy"`, and filter `fold=='test'`
for the ID dataset) and `src/paper_helpers/ood_metrics/runner.py::run_ood_comparison`
(thin wrapper around both).

## Inference-time CSVs (`src/inference/infer.py`)

Written by `ClassificationInferenceRunner`/`ArtifactInferenceRunner` to
`<save_path>/predictions.csv`:

| column | meaning |
|---|---|
| `image_id` | sample identifier |
| `fold` | source fold |
| `target` | ground-truth class index |
| `prediction` | predicted class index |
| `confidence` | max softmax probability -- **note: named `confidence`, not `prediction_prob_score`**. `score_mode="msp"` in `auc.py` accepts either name, callback's first |
| `class_logits` | JSON-encoded list, logits per class. Always written |
| `class_probs` | JSON-encoded list, softmax probabilities per class |
| `stream` | `"real"`/`"artifact"` in artifact mode, `"default"` otherwise. **Always present** -- never key schema detection on it |
| `raw_logits` | JSON-encoded list. Only when the net emits it (SNGP's pre-mean-field head output) |
| `predictive_entropy` | float, Shannon entropy of `class_probs` in nats. **Always written.** One of the three cross-family comparable uncertainty columns |
| `confidence_margin` | float, top-1 minus top-2 probability. **Always written.** Separates a confident row from a near-tie, which `confidence` alone cannot |
| `dempster_shafer` | float in `[0, 1]`, `K / (K + sum(exp(class_logits)))`. **Always written.** Larger = less total evidence |
| `uncertainty` | one scalar per sample. Only when the model exposes it. **Its unit differs per family — see the warning below.** Read `uncertainty_kind` before comparing it across runs |
| `uncertainty_kind` | string: `gp_predictive_variance` (SNGP), `member_logit_variance` (Deep Ensemble), `mc_logit_std` (MC-Dropout), or `unknown`. Written alongside `uncertainty` from schema 3 on |
| `total_entropy` | float, nats. `H[mean_m softmax(member_m)]` — total predictive uncertainty. Only when a member stack exists (ensemble or MC-Dropout) |
| `aleatoric_entropy` | float, nats. `mean_m H[softmax(member_m)]` — data uncertainty. Same availability as `total_entropy` |
| `mutual_information` | float, nats. `total_entropy - aleatoric_entropy` — epistemic (model) uncertainty, a.k.a. BALD. Same availability. **Written even when `save_member_logits=false`**, which is the point: the decomposition costs three floats, the raw stack costs ~4.8x a row |
| `member_logits` | JSON-encoded nested list, `[M, C]` per row. Only when `infer.save.save_member_logits=true` **and** the net has members -- ensemble members or MC-Dropout passes (same shape convention either way). Off by default: multiplies row size by roughly `M` |
| `count` | int. Artifact mode only, artifact-stream rows only. The pinned overlay count (`data.datamodule.artifact_count`) -- constant for the whole run, present only when set |
| `severity` | int, 1-5. Artifact mode only, artifact-stream rows only. The procedural severity grade (`data.datamodule.artifact_severity`) -- constant for the whole run, present only when set |
| `percent_pixels_affected` | float, 0-100. Artifact mode only, artifact-stream rows only. Per-sample fraction of pixels with nonzero alpha in the *localized* `artifact_mask` (mask is 0-255 blend alpha, not 0/1 -- thresholded at `>0`, not averaged, or a few near-opaque pixels would masquerade as low coverage) -- usually 0 on a procedural-only run, since most procedural effects are whole-frame; the exception is `local_blur` (the one procedural effect with local, not global/warp, scope), which localizes into the mask like a pasted overlay would. **0 here does not mean the image was untouched** -- see `global_degradations` below |
| `global_degradations` | JSON-encoded list of effect names, e.g. `["hed_stain_shift"]` or `[]`. Artifact mode only, artifact-stream rows only. Whole-frame photometric effects that fired (stain shift, noise, compression, brightness/contrast, illumination, vignette, chromatic aberration, defocus/motion blur, tile seam) -- these change every pixel's value but never touch `artifact_mask`, so this is what distinguishes "nothing happened" from "a global effect touched the whole image" when `percent_pixels_affected` reads 0 |
| `geometric_degradations` | JSON-encoded list, e.g. `["elastic_deformation"]` or `[]`. Artifact mode only, artifact-stream rows only. Kept separate from `global_degradations` because the simulator itself keeps warps apart from photometric effects -- a different failure mode, and merging the two would make either a false positive for the other in downstream filtering |

No `true_bin_label` — derive it as `prediction == target`.

**Artifact-mode sweeps write split files, not one dual-stream CSV.** The real/clean stream
is bit-identical across every artifact/procedural config against one checkpoint
(`data.datamodule.artifact_count`/`artifact_severity` don't touch `real_image`), so
resaving it on every swept run is pure duplication. Convention (see
`docs/DATASETS.md`'s "Saving sweep results" and the `inference` skill's `references/commands.md`):
a `real_baseline/predictions.csv` written once per checkpoint (`infer.save.streams=[real]`,
`stream` column all `"real"`), and each swept variant's own `predictions.csv`
(`infer.save.streams=[artifact]`, `stream` column all `"artifact"`, `count`/`severity`/
`percent_pixels_affected` populated). A pre-sweep, self-contained dual-stream CSV (the
`infer.save.streams` default, both streams in one file) is still the normal shape for a
one-off artifact-robustness check that isn't part of a sweep.

`class_logits` and `raw_logits` were added in `predictions_csv_schema: 2`; runs written
before that have neither, and logits cannot be reconstructed from `class_probs`
(softmax is shift-invariant), so a logits-based metric on an older run needs inference
re-run, not a converter.

`predictions_csv_schema: 3` added `predictive_entropy`, `confidence_margin`,
`dempster_shafer`, `uncertainty_kind`, and the
`total_entropy`/`aleatoric_entropy`/`mutual_information` decomposition. Same rule: no
converter, re-run inference.

### `uncertainty` is not comparable across model families

The column holds a **different physical quantity** depending on what produced it:

| family | `uncertainty_kind` | what it is |
|---|---|---|
| SNGP | `gp_predictive_variance` | GP posterior variance of the latent function, `phi^T Sigma phi`. Shared across classes, unbounded, data-scale-dependent |
| Deep Ensemble | `member_logit_variance` | mean over classes of the per-class **variance** of member logits |
| MC-Dropout | `mc_logit_std` | mean over classes of the per-class **standard deviation** of per-pass logits |
| Baseline (plain) | *(column absent)* | no model-side variance exists |

Three units and one absence. Ranking families by `uncertainty` is meaningless — use
`predictive_entropy` / `dempster_shafer` / `confidence_margin`, which are defined on
quantities every family emits, or `mutual_information` when comparing two runs that
both have member stacks. `src/metrics/uncertainty.py` is the single implementation
behind all of these, shared by both writers.

Note `total_entropy` can differ slightly from `predictive_entropy` on the same row for
a Deep Ensemble: the decomposition uses `mean(softmax(member_logits))`, the correct
predictive distribution for a model average, while `class_probs` comes from
`softmax(mean(member_logits))` (`DeepEnsemble.forward`'s logit-space pooling). That is
a real property of the two poolings, not an inconsistency.

**Caveat for `mc_*` runs**: `mc_predict` returns the mean of per-pass logits alongside
the mean of per-pass softmax, so `softmax(class_logits) != class_probs` there. That is
inherent to the estimator, not a bug.

**Consumed by**: `src/metrics/artifact_quantification.py::quantify_artifact_impact`
(requires `image_id, target, prediction, confidence, class_probs, stream`; canonicalizes
`stream` synonyms via `_canonicalize_stream`). Its optional `real_csv_map` param joins an
artifact-only sweep CSV against its checkpoint's `real_baseline` CSV before pairing --
needed for any model entry that came from a `infer.save.streams=[artifact]` sweep run.

## If you need one schema and have the other

Don't write a new one-off converter inline. Add a small, named conversion helper next
to whichever consumer needs it (or extend it if one already exists) — future callers
will need the same conversion.
