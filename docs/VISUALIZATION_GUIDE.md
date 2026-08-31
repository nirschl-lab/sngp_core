# Visualization Guide

Publication-figure scripts live in [src/visualization/](src/visualization/). Every
plotting function calls `set_default_style()` from
[src/visualization/style.py](src/visualization/style.py) first (shared
matplotlib/seaborn rcParams -- whitegrid, pastel palette, font sizes) and **returns a
`Figure`** rather than saving or calling `plt.show()` internally -- saving happens in
each script's own `main()`/`__main__` block, so the same plotting function also works
for training-time W&B logging (`src/callbacks/test_artifacts_callback.py` passes the
returned figure to `wandb.Image(fig)`).

Untested by policy (see `.claude/rules/testing.md`) -- smoke-test a new or changed
script manually against real data rather than expecting a pytest suite.

## Predictive Entropy

[src/visualization/predictive_entropy.py](src/visualization/predictive_entropy.py)
compares a model's predictive entropy (Shannon entropy in nats, computed from each
row's `class_probs`) across datasets: a KDE density overlay with a solid line for
in-distribution dataset(s) and dashed lines for out-of-distribution ones, so you can
see at a glance whether the model is more uncertain on OOD data.

It reads from the per-dataset inference output layout described in
[docs/OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md) --
`<run_dir>/<dataset_name>/predictions.csv` -- and expects `--indist`/`--outdist` to
name the dataset folders directly (not full paths):

```bash
uv run src/visualization/predictive_entropy.py \
  --run-dir /data1/maheswararao/experiments/uncertaity-aware-ml/infer/baseline_classifier_acevedo/2026-08-25_14-31-36 \
  --indist acevedo \
  --outdist jung kather2016 kather2018 nirschl2018 tang wong \
  --name baseline_acevedo_entropy
```

Notes:

- Sampling is class-balanced (`--n-total`, default 500 rows per dataset) so KDE
  comparisons stay fair across datasets of very different sizes.
- `class_probs` length (not the OOD dataset's own `num_classes`) determines the
  entropy calc -- OOD datasets are run through the in-distribution model's own output
  head.
- Output: `<figures-dir>/<name>.png` and `.pdf` (`--figures-dir` defaults to
  `figures/predictive_entropy/` at the repo root, matching the existing
  `figures/dataset_stats/`/`figures/auroc/` convention -- untracked, not committed).
