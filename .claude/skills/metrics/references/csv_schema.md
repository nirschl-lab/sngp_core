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
| `class_logits` | JSON-encoded list, raw logits per class |
| `class_probs` | JSON-encoded list, softmax probabilities per class |
| `fold` | `train`/`validation`/`test` |

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
| `class_probs` | JSON-encoded list, softmax probabilities per class |
| `stream` | `"real"` or `"artifact"` in artifact mode; absent/constant otherwise |
| `uncertainty` | present when the model exposes it (SNGP variance, MC-Dropout std) |

No `class_logits`, no `true_bin_label`.

**Consumed by**: `src/paper_helpers/ood_metrics/artifact_quantification.py::quantify_artifact_impact`
(requires `image_id, target, prediction, confidence, class_probs, stream`; canonicalizes
`stream` synonyms via `_canonicalize_stream`).

## If you need one schema and have the other

Don't write a new one-off converter inline. Add a small, named conversion helper next
to whichever consumer needs it (or extend it if one already exists) — future callers
will need the same conversion.
