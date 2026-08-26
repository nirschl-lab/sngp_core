# CLAUDE.md

Uncertainty-aware histopathology image classification research (`sngp-core`), built on
PyTorch Lightning + Hydra (fork of `ashleve/lightning-hydra-template`). Three model
families — Baseline, SNGP, Deep Ensemble — share resnet/ViT backbones across 7
HuggingFace datasets under the `nirschl-lab` org. The paper (ISBI 2026,
arXiv:2602.02370) is accepted; ongoing work evaluates robustness to simulated imaging
artifacts and publishes trained models to the HF Hub.

## Where things live

This file is deliberately short. Project knowledge and engineering rules live outside
it, written to stand on their own for any contributor — human or not:

- **`docs/`** — what the project is, how it's architected, and how to do things in it.
  Start at [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md); it links out to
  [docs/DATASETS.md](docs/DATASETS.md),
  [docs/SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md),
  [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md), and the topic guides (inference, HPO,
  deep ensembles, output layout).
- **`.claude/rules/`** — must-follow engineering rules (environment, hard contracts,
  testing, conventions, git workflow), one topic per file. Imported below so they're
  always in context for this session.

@.claude/rules/environment.md
@.claude/rules/hard-contracts.md
@.claude/rules/testing.md
@.claude/rules/conventions.md
@.claude/rules/git-workflow.md

## Claude Code shortcuts

[docs/DEVELOPMENT.md#extending-the-framework](docs/DEVELOPMENT.md#extending-the-framework)
lists what each workflow requires manually. In a Claude Code session, prefer these
skills over doing it by hand:

| Task | Skill |
|---|---|
| New DataModule variant (new loading/pairing/filtering logic, not just a new dataset) | `add-datamodule` |
| New backbone / net family | `add-model` |
| New training strategy (Lightning module) | `add-lightning-module` |
| Run inference / checkpoint sweeps / artifact inference | `inference` |
| Offline/research metrics (AUROC-OOD, calibration, artifact quantification) | `metrics` |
| Publication figures | `visualizations` |

A plain new dataset with the existing schema is config-only
([docs/DATASETS.md#adding-a-dataset](docs/DATASETS.md#adding-a-dataset)) — no skill
needed.
