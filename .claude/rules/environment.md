# Environment & tooling

**uv only.** Never run bare `python` or `pip install` — always `uv run <script>` /
`uv sync`. `requirements.txt`/`environment.yaml` exist for legacy reasons only;
`uv.lock`/`pyproject.toml` are authoritative.

Requires a `.env` file (copy from `env_example`) with a `PROJECT_ROOT` variable —
`configs/paths/default.yaml` resolves `root_dir` from it. `log_dir`,
`feature_cache_dir`, `data_cache_dir` in that same config point at absolute cluster
paths under `/data1/...` — machine-specific, not portable, and not controlled by
`.env`.

Full setup walkthrough (uv install, `wandb login`, `huggingface-cli login`, command
reference):
[docs/DEVELOPMENT.md#environment-setup](../../docs/DEVELOPMENT.md#environment-setup).
