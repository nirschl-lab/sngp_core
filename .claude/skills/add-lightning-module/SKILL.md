---
name: add-lightning-module
description: Add a new PyTorch Lightning training strategy (LightningModule) for this project -- a new loss regime, a new uncertainty method's training loop, or a multi-stage training scheme like Deep Ensembles. Use when the training/validation/test step logic itself needs to differ, not just the net architecture.
---

# Add a Lightning training module

Scope: a new `LightningModule` subclassing `LitModuleBase` — used when *how training
happens* differs (loss computation, multi-stage/multi-member training, custom logged
terms), not just the net architecture. If only the net differs, use the `add-model`
skill and pair it with an existing LitModule (`BaselineClassificationLitModule` or
`SNGPClassificationLitModule`).

## Ask first

1. What differs from `LitModuleBase`'s default `model_step`/`forward`? (loss
   computation, extra logged metrics, multi-stage training like ensemble member
   cycling, custom `configure_optimizers` behavior)
2. Which net(s) does it pair with — an existing one, or a new one from `add-model`?
3. New hyperparameters this module needs (with sensible defaults).
4. Does it need custom checkpoint state beyond what `LitModuleBase` already tracks
   (net spec, num_classes, etc.)?

## Reference implementations (read before writing)

- `src/models/lit_module_base.py` — shared base: `training_step`/`validation_step`/
  `test_step` call `model_step` (the method subclasses override); metric tracking;
  `configure_optimizers`; the checkpoint hooks (`on_save_checkpoint`/`on_load_checkpoint`).
- `src/models/sngp_classification_lit_module.py` — the minimal-override reference: only
  `forward` and `model_step` differ from the base.
- `src/models/ensemble/deep_ensemble_lit_module.py` — the maximal-override reference:
  multi-stage training (`on_train_epoch_start` cycles the active member),
  `configure_optimizers` override, `test_step` override for ensemble-specific
  uncertainty. Useful as a template if the new strategy needs epoch-level state.
- `src/checkpointing/spec.py` — `build_meta`, the checkpoint metadata contract.

## The `model_step` contract

Every `model_step` implementation must return the same 7-tuple, in this order:

```python
return img_ids, loss, logits, probs, preds, targets, fold
```

`training_step`/`validation_step`/`test_step` in `LitModuleBase` all unpack this
tuple; changing its shape breaks every subclass, not just yours.

## Steps

1. Create `src/models/<name>_lit_module.py`:
   ```python
   from typing import Optional
   import torch
   from src.models.lit_module_base import LitModuleBase

   class MyLitModule(LitModuleBase):
       def __init__(
           self,
           net: Optional[torch.nn.Module] = None,
           optimizer: Optional[torch.optim.Optimizer] = None,
           scheduler: torch.optim.lr_scheduler = None,
           compile: bool = False,
           my_new_param: float = 0.1,
           **kwargs,
       ) -> None:
           super().__init__(net=net, optimizer=optimizer, scheduler=scheduler, compile=compile, **kwargs)
           self.my_new_param = my_new_param

       def model_step(self, batch):
           img_ids, x, targets, fold = batch
           logits = self.forward(x).logits
           probs = torch.softmax(logits, dim=1)
           loss = self.criterion(logits, targets)  # or your custom loss
           preds = torch.argmax(probs, dim=1)
           return img_ids, loss, logits, probs, preds, targets, fold
   ```

2. Add `configs/model/<name>.yaml` (see `configs/model/baseline_classifier.yaml` for
   the shape; include `defaults: [- calibration@calibration_cfg: default, - _self_]`
   only if the module accepts `calibration_cfg`).

3. Tests, `tests/models/test_<name>_lit_module.py`:
   - **Non-negotiable**: extend the parametrization in
     `tests/checkpointing/test_hparams_are_primitive.py` (or add an equivalent test)
     to cover the new module — this is the permanent regression guard for the
     checkpoint-portability bug this project's migration fixed. Any new non-primitive
     constructor argument (a class, a partial, a config object) MUST be added to the
     `ignore=[...]` list in `LitModuleBase.save_hyperparameters()` — or, if it's
     specific to your subclass, ignored the same way `calibration_cfg` is (see
     `LitModuleBase.__init__`'s `save_hyperparameters(ignore=[...])` call — the
     ignore list applies across the whole `__init__` call chain, not just the frame
     it's declared in).
   - A `fast_dev_run`-style smoke test: instantiate via Hydra, run one training step
     on a tiny fake batch, assert it doesn't error and produces a finite loss (see
     `tests/checkpointing/test_roundtrip.py::_build_and_train_one_step` for the
     pattern — reuse it rather than re-deriving).

## Hard checklist

- [ ] `net`, `optimizer`, `scheduler`, and any other non-primitive constructor arg
      have `Optional[...] = None` defaults (needed so `cls(**hparams)` — used by
      checkpoint reconstruction — works without them).
- [ ] `model_step` returns the 7-tuple in the exact order above.
- [ ] No wandb-specific (or any logger-specific) code in the module — test-time
      CSV/figure logging belongs in `src/callbacks/test_artifacts_callback.py`
      (extend it if you need new artifacts), not in the LightningModule. `self.log(...)`
      is fine anywhere; `self.logger.experiment.log(...)` is not.
- [ ] New hparams checked with `json.dumps(dict(model.hparams))` — must not raise.

## Verify

```bash
uv run pytest tests/models -k <name>
uv run pytest tests/checkpointing
uv run src/train.py experiment=<existing_experiment> model=<name> trainer.fast_dev_run=true
```
