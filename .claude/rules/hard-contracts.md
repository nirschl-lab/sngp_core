# Hard contracts: nets and checkpoints

Every net and every checkpoint in this project satisfies these. Check new code
against them before anything else — see
[docs/DEVELOPMENT.md#model-methodology](../../docs/DEVELOPMENT.md#model-methodology)
for the architectural reasoning behind them.

- **Every net's `forward()` returns a `ModelOutput`** (`src/models/outputs.py`) —
  `.logits` for loss/argmax, `.variance` for uncertainty (SNGP predictive variance /
  ensemble disagreement), `.raw_logits`/`.features`/`.member_logits` where relevant.
  Never a bare tensor, never a bespoke tuple.
- **Every net has a `.spec` property** (plain JSON-serializable dict: registry `name` +
  ctor kwargs) and is registered in `NET_REGISTRY` via `@register_net("...")`. Backbone
  construction lives in exactly one place, `src/models/backbones.py`.
- **`LitModuleBase.save_hyperparameters()` ignores `net`/`optimizer`/`scheduler`**
  — none of those are JSON-serializable. `checkpoint["hyper_parameters"]`
  must always be JSON-serializable; `tests/checkpointing/test_hparams_are_primitive.py`
  is a permanent regression guard for this. This is the fix for the historical
  "checkpoints expect the same code structure they were saved with" problem: the old
  code pickled the live `net` object into hparams, so unpickling required the
  *original* class's import path to still resolve.
- **All checkpoint I/O goes through `src/checkpointing/io.py`** — never raw
  `torch.load` + manual state-dict surgery, never a bare
  `LightningModule.load_from_checkpoint()` scattered across scripts. `read_meta()`
  tells you a checkpoint's architecture/format without unpickling anything.
- **Checkpoints below `FORMAT_VERSION` are refused**, not silently degraded. Migrate
  once with `scripts/checkpoints/migrate_checkpoints.py`; there is no dual-format
  reading in production code.
