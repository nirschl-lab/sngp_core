---
name: add-lightning-module
description: Add a new PyTorch Lightning training strategy (LightningModule) for this project -- a new loss regime, a new uncertainty method's training loop, or a multi-stage training scheme like Deep Ensembles. Use when the training/validation/test step logic itself needs to differ, not just the net architecture.
---

# Add a Lightning training module

Scope: a new `LightningModule` subclassing `LitModuleBase` — used when *how training
happens* differs (loss computation, multi-stage/multi-member training, custom logged
terms), not just the net architecture. If only the net differs, use the `add-model`
skill and pair it with an existing LitModule (`BaselineLitModule` or `SNGPLitModule`).

## Ask first

1. What differs from `LitModuleBase`'s default `model_step`/`forward`? (loss
   computation, extra logged metrics, multi-stage training like ensemble member
   cycling, custom `configure_optimizers` behavior)
2. Which net(s) does it pair with — an existing one, or a new one from `add-model`?
3. New hyperparameters this module needs (with sensible defaults).
4. Does it need custom checkpoint state beyond what `LitModuleBase` already tracks
   (net spec, num_classes, etc.)?

## Reference implementations (read before writing)

- `src/models/lit_module_base.py` — shared base: a lean train/val loop only.
  `training_step`/`validation_step` call `model_step` (the method subclasses
  override for a non-CE training loss); metric tracking (`train_acc`, `val_metrics`
  `MetricCollection`); `configure_optimizers`; the checkpoint hooks
  (`on_save_checkpoint`/`on_load_checkpoint`). `model_step` already implements plain
  CE classification, shared by every family unless yours genuinely needs something
  different -- most new modules need **no** `model_step` override at all.
- `src/models/sngp_lit_module.py` — the minimal-override reference: only `forward`
  differs from the base (no `model_step` override).
- `src/models/baseline_lit_module.py` — the test/predict-only-override reference:
  `_predict_forward` swaps in MC-Dropout averaging without touching train/val.
- `src/models/deep_ensemble_lit_module.py` — the multi-stage-training reference:
  `on_train_epoch_start` cycles the active member epoch-by-epoch,
  `on_validation_epoch_end` logs ensemble-specific progress. Otherwise thin --
  test-time uncertainty (`ModelOutput.variance`/`.member_logits`) is handled
  generically by the shared `test_step`, no override needed.
- `src/checkpointing/spec.py` — `build_meta`, the checkpoint metadata contract.

## The `model_step` contract (train/val only)

Every `model_step` implementation must return the same 7-tuple, in this order:

```python
return img_ids, loss, logits, probs, preds, targets, fold
```

`training_step`/`validation_step` in `LitModuleBase` unpack this tuple; changing its
shape breaks every subclass, not just yours. **`test_step`/`predict_step` do not call
`model_step`** -- they call `self._predict_forward(x)` (defaults to `self.forward(x)`;
override this instead if your family needs different behavior only at test/predict
time, e.g. MC-Dropout averaging) and return a dict of raw outputs for
`src/callbacks/test_artifacts_callback.py::TestArtifactsCallback` to accumulate and
analyze. Never add loss computation, metric `.update()` calls, or CSV/figure logic to
`test_step` -- that all belongs in the callback.

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

       # Only override model_step if your training loss genuinely isn't plain CE --
       # otherwise skip this method entirely and inherit LitModuleBase.model_step.
       def model_step(self, batch):
           img_ids, x, targets, fold = batch
           logits = self.forward(x).logits
           probs = torch.softmax(logits, dim=1)
           loss = self.criterion(logits, targets)  # or your custom loss
           preds = torch.argmax(probs, dim=1)
           return img_ids, loss, logits, probs, preds, targets, fold
   ```

2. Add `configs/model/<name>.yaml` (see `configs/model/baseline_classifier.yaml` for the shape).

3. Tests, `tests/models/test_<name>_lit_module.py`:
   - **Non-negotiable**: extend the parametrization in
     `tests/checkpointing/test_hparams_are_primitive.py` (or add an equivalent test)
     to cover the new module — this is the permanent regression guard for the
     checkpoint-portability bug this project's migration fixed. Any new non-primitive
     constructor argument (a class, a partial, a config object) MUST be added to the
     `ignore=[...]` list in `LitModuleBase.save_hyperparameters()` (currently just
     `net`/`optimizer`/`scheduler`).
   - A `fast_dev_run`-style smoke test: instantiate via Hydra, run one training step
     on a tiny fake batch, assert it doesn't error and produces a finite loss (see
     `tests/checkpointing/test_roundtrip.py::_build_and_train_one_step` for the
     pattern — reuse it rather than re-deriving).

## Hard checklist

- [ ] `net`, `optimizer`, `scheduler`, and any other non-primitive constructor arg
      have `Optional[...] = None` defaults (needed so `cls(**hparams)` — used by
      checkpoint reconstruction — works without them).
- [ ] `model_step` (if overridden) returns the 7-tuple in the exact order above.
- [ ] `test_step`/`predict_step` are not overridden unless `_predict_forward` genuinely
      can't express the needed test-time behavior. If you do override `test_step`,
      keep the same return dict shape: `img_ids, fold, logits, probs, preds, targets,
      variance, member_logits` (the last two `None` if not applicable) — that's the
      contract `TestArtifactsCallback` expects via `on_test_batch_end`.
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
