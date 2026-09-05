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

No `true_bin_label` — derive it as `prediction == target`.

`class_logits` and `raw_logits` were added in `predictions_csv_schema: 2`; runs written
before that have neither, and logits cannot be reconstructed from `class_probs`
(softmax is shift-invariant), so a logits-based metric on an older run needs inference
re-run, not a converter.

**Caveat for `mc_*` runs**: `mc_predict` returns the mean of per-pass logits alongside
the mean of per-pass softmax, so `softmax(class_logits) != class_probs` there. That is
inherent to the estimator, not a bug.

**Consumed by**: `src/paper_helpers/ood_metrics/artifact_quantification.py::quantify_artifact_impact`
(requires `image_id, target, prediction, confidence, class_probs, stream`; canonicalizes
`stream` synonyms via `_canonicalize_stream`).

## If you need one schema and have the other

Don't write a new one-off converter inline. Add a small, named conversion helper next
to whichever consumer needs it (or extend it if one already exists) — future callers
will need the same conversion.
