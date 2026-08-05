# histo-artifact-sim

A minimal starter package for reproducible histopathology artifact simulation.
It supports:

- RGBA artifact assets where alpha is the artifact mask.
- Separate artifact image and mask pairs.
- Category-specific occurrence and coverage distributions in YAML.
- Normal, multiply, screen, and darken blending.
- Procedural artifacts: illumination gradients, local blur, pixelation, and brightness/contrast shifts.
- A callable API for custom/PyTorch datasets.
- Output artifact masks and JSON metadata for every simulation.

## Design principle

Keep three concerns separate:

1. **Assets**: a CSV manifest that says where each artifact lives and what category it belongs to.
2. **Simulation policy**: YAML probabilities, coverage distributions, opacities, and procedural effects.
3. **Execution**: a pipeline used either in a data loader or from the command line.

Adding a new artifact type normally requires only adding assets and a YAML entry, not modifying the pipeline.

## Recommended asset layout

```text
assets/
  alpha/
    debris/
      debris_001.png          # RGBA
    pigment_ink/
      ink_001.png
  paired/
    bubble/
      images/
        bubble_001.png
      masks/
        bubble_001.png
```

An unstructured layout is also usable when `--taxonomy-csv` maps `image_name` to `parent_category` and `sub_category`.

## Install

**Development (this repo):**
```bash
python -m pip install -e . --no-build-isolation
```

## Using this package in another project

### Option A — install from Git (pip)

```bash
pip install "git+ssh://git@git.doit.wisc.edu/smph/path/nirschl-lab/histo-artifact-sim.git@main"
```

To pull new changes later:
```bash
pip install --upgrade "git+ssh://git@git.doit.wisc.edu/smph/path/nirschl-lab/histo-artifact-sim.git@main"
```

### Option A — install from Git (uv)

```bash
uv add "git+ssh://git@git.doit.wisc.edu/smph/path/nirschl-lab/histo-artifact-sim.git@main"
```

To pull new changes later:
```bash
uv lock --upgrade-package histo-artifact-sim && uv sync
```

### Option B — editable local clone (live development)

Clone once, then install as editable so source edits are reflected immediately:

```bash
git clone git@git.doit.wisc.edu:smph/path/nirschl-lab/histo-artifact-sim.git /path/to/local/clone
pip install -e /path/to/local/clone
# or with uv:
uv add -e /path/to/local/clone
```

To get new changes: `git pull` inside the clone — no reinstall needed.

## 1. Build a manifest

```bash
histo-artifacts index \
  --alpha-root /path/to/alpha_assets \
  --paired-root /path/to/paired_assets \
  --taxonomy-csv /path/to/img_names_categorized.csv \
  --output artifacts.csv
```

The manifest schema is:

```text
asset_path,mask_path,source_type,parent_category,sub_category
```

`source_type` is either `rgba` or `image_mask`.

## 2. Simulate a folder

```bash
histo-artifacts simulate \
  --input /path/to/source_patches \
  --manifest artifacts.csv \
  --config configs/empirical.yaml \
  --output simulated \
  --seed 42
```

The output contains simulated images, masks, and `simulation_metadata.jsonl`.

## 3. Use in a data loader

```python
from histo_artifacts import ArtifactPipeline

simulator = ArtifactPipeline.from_files(
    "artifacts.csv",
    "configs/balanced.yaml",
    seed=42,
)

class Dataset:
    def __getitem__(self, index):
        image, target = load_sample(index)
        simulated = simulator(image)
        return {
            "image": simulated["image"],
            "target": target,
            "artifact_mask": simulated["artifact_mask"],
            "artifact_metadata": simulated["metadata"],
        }
```

For multi-worker PyTorch loading, pass a per-sample seed derived from the worker seed and index:

```python
result = simulator(image, seed=worker_seed + index)
```

## Choosing a sampling profile

- `configs/empirical.yaml` uses the current observed category shares and coverage summaries as a starting policy, including the high debris frequency.
- `configs/balanced.yaml` gives categories equal selection probability while preserving their category-specific size distributions.

For model robustness work, balanced sampling is often more useful than copying the asset collection's raw frequency. Keep a clean-image probability and report performance by category and severity.

## First extensions to add later

- Tissue-aware placement using a tissue mask.
- Multiple instances until a requested total coverage is reached.
- Stain-space effects in HED/optical-density space.
- Elastic deformation for fibers/tissue folds.
- A validation report comparing simulated and real category/coverage distributions.
