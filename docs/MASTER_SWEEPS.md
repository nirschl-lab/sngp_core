# Master list of W&B sweeps

One bullet per sweep, appended automatically by `scripts/hpo/sweep.py` (pilots run
with `--no-register` are not listed). Read results with
`uv run scripts/hpo/summarize_sweep.py --sweep <path>`; protocol in docs/HPO_GUIDE.md.
- `acevedo_baseline_resnet18_hpo`: `nirschl-lab/uncertainty-aware-ml/ogw7zwgt` (created 2026-09-16, objective `val/nll_cal_best` minimize, run_cap 32, git 58dadba)
- `acevedo_sngp_resnet18_hpo`: `nirschl-lab/uncertainty-aware-ml/r5mjo378` (created 2026-09-16, objective `val/nll_cal_best` minimize, run_cap 48, git 58dadba)
- `adrc_wong_sngp`: `nirschl-lab/uncertainty-aware-ml/5ua75jv5` (created 2026-10-04, objective `val/nll_best` minimize, grid 30, git c988225)
- `adrc_wong_sngp_bnsn`: `nirschl-lab/uncertainty-aware-ml/i46c70y1` (created 2026-10-04, objective `val/nll_best` minimize, grid 30, git c988225)
- `adrc_wong_specreg`: `nirschl-lab/uncertainty-aware-ml/nznple49` (created 2026-10-04, objective `val/nll_best` minimize, grid 30, git c988225)
- `adrc_wong_muon`: `nirschl-lab/uncertainty-aware-ml/3yx5yktr` (created 2026-10-04, objective `val/nll_best` minimize, grid 30, git c988225)
