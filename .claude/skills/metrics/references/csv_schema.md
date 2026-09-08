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

The two list columns here are written as Python list objects and stringified by pandas
(`repr`, not JSON) — which is why `_parse_class_probs` needs its `ast.literal_eval`
fallback. The inference schema below uses real `json.dumps` instead.

**Consumed by**: `src/metrics/auc.py::AUROC_across_dataset` (reads
`prediction_prob_score` for `score_mode="msp"`, `class_probs` for
`score_mode="entropy"`, filters `fold=='test'` for the ID dataset) and
`src/paper_helpers/ood_metrics/runner.py::run_ood_comparison` (thin wrapper around it).

## Inference-time CSVs (`src/inference/infer.py`)

Written by `ClassificationInferenceRunner`/`ArtifactInferenceRunner` to
`<save_path>/predictions.csv`:

| column | meaning |
|---|---|
| `image_id` | sample identifier |
| `fold` | source fold |
| `target` | ground-truth class index |
| `prediction` | predicted class index |
| `confidence` | max softmax probability -- **note: named `confidence`, not `prediction_prob_score`** |
| `class_logits` | JSON-encoded list, logits per class. Always written |
| `class_probs` | JSON-encoded list, softmax probabilities per class |
| `stream` | `"real"`/`"artifact"` in artifact mode, `"default"` otherwise. **Always present** -- never key schema detection on it |
| `raw_logits` | JSON-encoded list. Only when the net emits it (SNGP's pre-mean-field head output) |
| `uncertainty` | one scalar per sample. Only when the model exposes it (SNGP/ensemble variance, MC-Dropout std) |
| `member_logits` | JSON-encoded nested list, `[M, C]` per row. Only when `infer.save.save_member_logits=true` **and** the net has members -- ensemble members or MC-Dropout passes (same shape convention either way). Off by default: multiplies row size by roughly `M` |
| `count` | int. Artifact mode only, artifact-stream rows only. The pinned overlay count (`data.datamodule.artifact_count`) -- constant for the whole run, present only when set |
| `severity` | int, 1-5. Artifact mode only, artifact-stream rows only. The procedural severity grade (`data.datamodule.artifact_severity`) -- constant for the whole run, present only when set |
| `percent_pixels_affected` | float, 0-100. Artifact mode only, artifact-stream rows only. Per-sample fraction of pixels with nonzero alpha in the *localized* `artifact_mask` (mask is 0-255 blend alpha, not 0/1 -- thresholded at `>0`, not averaged, or a few near-opaque pixels would masquerade as low coverage) -- usually 0 on a procedural-only run, since most procedural effects are whole-frame; the exception is `local_blur` (the one procedural effect with local, not global/warp, scope), which localizes into the mask like a pasted overlay would |

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

**Caveat for `mc_*` runs**: `mc_predict` returns the mean of per-pass logits alongside
the mean of per-pass softmax, so `softmax(class_logits) != class_probs` there. That is
inherent to the estimator, not a bug.

**Consumed by**: `src/paper_helpers/ood_metrics/artifact_quantification.py::quantify_artifact_impact`
(requires `image_id, target, prediction, confidence, class_probs, stream`; canonicalizes
`stream` synonyms via `_canonicalize_stream`). Its optional `real_csv_map` param joins an
artifact-only sweep CSV against its checkpoint's `real_baseline` CSV before pairing --
needed for any model entry that came from a `infer.save.streams=[artifact]` sweep run.

## If you need one schema and have the other

Don't write a new one-off converter inline. Add a small, named conversion helper next
to whichever consumer needs it (or extend it if one already exists) — future callers
will need the same conversion.
