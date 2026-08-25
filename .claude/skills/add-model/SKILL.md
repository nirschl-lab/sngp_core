---
name: add-model
description: Add a new backbone, net class, or net family to this project (resnet/vit-style classifiers, SNGP variants, ensembles). Use when the user wants a new architecture, a new backbone option, or a new uncertainty-quantification net wired into training/inference/checkpointing.
---

# Add a model

Scope: a new **net** (an `nn.Module` passed as `LitModuleBase.net`) — either a new
backbone added to the shared factory, or an entirely new net class/family. For a new
*training strategy* around an existing net (a new loss, a new `model_step`), use the
`add-lightning-module` skill instead.

## When NOT to use this

- Changing an existing net's hyperparameter defaults — just edit its `configs/model/*.yaml`.
- Adding a dataset — that's config-only, see CLAUDE.md §5, no net changes needed.

## Ask first

1. **New backbone** (add e.g. `convnext_tiny` to the existing families) vs. **new net
   family** (a genuinely new architecture) vs. **variant of an existing family**?
2. Registry key / name (snake_case, e.g. `"my_classifier"`).
3. Which backbones must it support? Does it reuse `src/models/backbones.py`, or does
   it not use a torchvision backbone at all (e.g. a from-scratch architecture)?
4. Is it SNGP/spectral-norm compatible? (Only resnet-family backbones are today — see
   `SPECTRAL_NORM_COMPATIBLE` in `src/models/backbones.py`. If the new backbone is a
   transformer/attention-based architecture, assume **not** compatible unless the user
   says otherwise, and wire the guard accordingly.)
5. Constructor hyperparameters and defaults.
6. Does it produce uncertainty (`variance`) or per-member logits (`member_logits`), or
   just `logits`?
7. Should it be exportable to HuggingFace (`scripts/hf/export_to_hub.py`)? If yes, it
   needs to stay dependency-light (no `hydra`/`lightning`/other `src.*` imports beyond
   `src.models.{backbones,outputs,registry}`) — see `references/hf_export_notes.md`.

## Reference implementations (read before writing)

- `src/models/backbones.py` — the ONE place backbone construction happens. If adding
  a backbone, add an entry to `BACKBONES` here, not inside a net class.
- `src/models/outputs.py` — `ModelOutput`, the contract every net's `forward()` must
  return.
- `src/models/registry.py` — `register_net`, `build_net`.
- `src/models/baseline/baseline_models.py` — canonical minimal example: one backbone,
  one linear head, a `.spec` property, MC-Dropout helper.
- `src/models/sngp/sngp_classifier.py` — example of a net with extra structure (RFF-GP
  head) and a compatibility guard (`assert_spectral_norm_compatible`).
- `configs/model/baseline_classifier.yaml` — the Hydra config shape a new
  `configs/model/<name>.yaml` should follow.

## Steps

1. **New backbone into an existing family**: add an entry to `BACKBONES` in
   `src/models/backbones.py` (ctor, weights enum name, default weight attr, kind —
   `"resnet"` or `"vit"`, or a new `kind` if it's structurally different). If it's
   spectral-norm compatible, it's picked up automatically by
   `SPECTRAL_NORM_COMPATIBLE` (derived from `kind == "resnet"` today — extend that
   logic if the new backbone needs its own rule). Add it to the parametrized test in
   `tests/models/test_backbone_factory.py`.

2. **New net family**: create `src/models/<family>/<name>.py`:
   ```python
   from src.models.backbones import build_backbone
   from src.models.outputs import ModelOutput
   from src.models.registry import register_net

   @register_net("<registry_key>")
   class MyNet(nn.Module):
       def __init__(self, arch="resnet18", num_classes=2, ..., pretrained=True):
           super().__init__()
           self.feature_extractor, feat_dim = build_backbone(arch, pretrained)
           # ... your head ...

       @property
       def spec(self) -> dict:
           return {"name": self.registry_name, "arch": self.arch, "num_classes": self.num_classes, ...}

       def forward(self, x) -> ModelOutput:
           ...
           return ModelOutput(logits=logits, variance=variance)  # only set what applies
   ```
   Every constructor kwarg that appears in `spec` must be a plain primitive (or a
   plain dict/list of primitives) — this is what makes checkpoints portable across
   code changes (see CLAUDE.md §3 "Hard contracts"). Never store a live object,
   class, or callable in `spec`.

3. Add `configs/model/<name>.yaml` mirroring `baseline_classifier.yaml`'s shape
   (`_target_` pointing at a *LightningModule* — pair with an existing one like
   `BaselineLitModule` if the training loop doesn't need to change, or see
   `add-lightning-module` if it does).

4. Tests (`tests/models/<family>/test_<name>.py` or add to
   `tests/models/test_output_contract.py`):
   - Shape test: forward pass at each supported arch, `pretrained=False`, assert
     `ModelOutput.logits.shape == (B, num_classes)`.
   - Spec round-trip: `build_net(net.spec)` produces an equivalent net (see
     `tests/models/test_registry.py` for the pattern).
   - If SNGP-adjacent (spectral norm): a construction-time guard test, see
     `tests/models/sngp/test_spectral_norm.py::TestSpectralNormCompatibility`.

## Common mistakes

- **Don't** re-implement backbone construction (weights-enum lookup, ctor maps)
  inside the new net class — always call `build_backbone`. This exact duplication
  (4 copies) is what Stage 1 of this project's migration eliminated.
- **Don't** return a bare tensor or a bespoke tuple from `forward()` — always
  `ModelOutput`.
- **Don't** add a `_target_` in a Hydra config without a matching `@register_net(...)`
  — the registry key is what checkpoints depend on being stable, not the Python
  import path (which can move freely once registered).
- **Don't** put a live object (a class, a partial, an already-instantiated
  sub-module) in `.spec` — if you find yourself doing this, the net probably needs a
  nested spec instead (see `DeepEnsemble.spec`'s `base_model_spec` for the pattern of
  one net wrapping another).

## Verify

```bash
uv run pytest tests/models -k <name>
uv run src/train.py experiment=<existing_experiment> model=<name> trainer.fast_dev_run=true model.net.pretrained=false
```
