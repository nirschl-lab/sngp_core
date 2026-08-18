# Examples

Practical scripts showing how to load a trained checkpoint and run predictions. All of
these use the project's real, currently-maintained API:

- `src/checkpointing/io.py::load_net` -- the single canonical checkpoint loader
- `src/inference/predict_image.py` -- a thin single/batch-image wrapper around it
- `src/models/hf_loader.py::HFModelLoader` -- loading a published HF Hub model

For dataset-wide inference with metrics (accuracy/ECE/precision/recall/...) against
labeled data, use the Hydra entrypoint `src/inference/infer.py` instead of anything
here -- see [docs/INFERENCE_GUIDE.md](../docs/INFERENCE_GUIDE.md). There is
deliberately one entrypoint for that; these examples are for quick, unlabeled,
script-level use.

## Scripts

### `01_basic_inference.py`

Single-image prediction, batch prediction over a directory, and MC-Dropout
uncertainty estimation.

```python
from src.inference.predict_image import predict_image

result = predict_image("path/to/checkpoint.ckpt", "path/to/image.jpg")
print(result["predicted_class"], result["confidence"])
```

### `02_advanced_inference.py`

Loading a published HF Hub model, config-driven inference, averaging predictions
across multiple independently-trained checkpoints, and dataset-wide inference with
saved JSON output (unlabeled images; use `infer.py` if you have labels and want
metrics).

### `03_deep_ensemble_inference.py`

Deep Ensemble inference with uncertainty quantification (variance/entropy/mutual
information) and per-member predictions:

```bash
uv run examples/03_deep_ensemble_inference.py \
    --checkpoint path/to/deep_ensemble.ckpt \
    --image path/to/image.jpg \
    --class-names class1 class2 class3 \
    --uncertainty-type variance \
    --save-viz output/visualization.png
```

See [docs/DEEP_ENSEMBLES_GUIDE.md](../docs/DEEP_ENSEMBLES_GUIDE.md) for the full guide.

## Running

Each script's example functions are commented out under `if __name__ == "__main__"`;
update the placeholder checkpoint/image paths at the top of the function you want,
uncomment it, then run with `uv run`:

```bash
uv run examples/01_basic_inference.py
uv run examples/02_advanced_inference.py
```

## Model requirements

Every net in this project (baseline, SNGP, deep ensemble) returns a `ModelOutput`
dataclass from `forward()` -- use `.logits` for the class scores. See
`src/models/outputs.py`.

## Troubleshooting

**"Model does not have `mc_predict` method"** -- MC-Dropout uncertainty
(`use_mc_dropout=True`) is only implemented for `BaselineClassifier`. SNGP exposes
predictive variance directly (`net(x).variance`); Deep Ensemble exposes
`net.get_predictive_uncertainty()` -- see `examples/03_deep_ensemble_inference.py`.

**Device mismatch errors** -- pass `device="cpu"` (or `"cuda"`) explicitly to
`predict_image`/`predict_batch`/`load_net` rather than mixing devices manually.

**Checkpoint won't load / format_version error** -- checkpoints saved before this
project's checkpoint-contract rewrite need a one-time migration; see
`scripts/checkpoints/migrate_checkpoints.py`.
