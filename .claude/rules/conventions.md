# Coding conventions

- Logging: `loguru` / the project's `RankedLogger` in Lightning code, not bare
  `print`.
- Type hints on public functions.
- No new top-level output directories at repo root without discussion — see
  [docs/KNOWN_ISSUES.md](../../docs/KNOWN_ISSUES.md) for why the existing ~14 aren't
  being consolidated yet.
- Figures go through `src/visualization/style.py` for consistent styling.
- Never edit or import from `notebooks/archive/` — it's frozen, superseded
  exploration (see [docs/KNOWN_ISSUES.md](../../docs/KNOWN_ISSUES.md)).
- Absolute paths belong in Hydra configs (`configs/paths/`), not hardcoded in source.
- Every new net registers via `@register_net(...)` and returns `ModelOutput`; every
  new LightningModule keeps `net`/`optimizer`/`scheduler`/anything non-primitive out
  of `save_hyperparameters()` — see [hard-contracts.md](hard-contracts.md).
