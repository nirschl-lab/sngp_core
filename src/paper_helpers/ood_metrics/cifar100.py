"""CIFAR-100 (in-distribution) vs CIFAR-10 / SVHN (OOD) AUROC across the four
WideResNet-28-10 benchmark arms. Config-only call site; the comparison loop itself lives
in src/paper_helpers/ood_metrics/runner.py.

Emits **two tables**, mirroring the paper itself: MSP is the score its main-text tables
report (Table 3), Dempster-Shafer the one its appendix compares against (Table 11).

MSP — `max_k softmax(g_k(x))` — is the protocol number, directly comparable to the
published CIFAR-100 rows. Dempster-Shafer — `K / (K + Σ exp(logit))` — is kept as a
companion rather than dropped: softmax is shift-invariant, so MSP discards the total
logit magnitude an SNGP head is trained to modulate, and DS is the score the reference
*implementation* uses for CIFAR OOD detection (`dempster_shafer_ood` in
google/uncertainty-baselines `baselines/cifar/sngp.py`).

Both use `estimator="full_population"`: the whole CIFAR-100 test set against the whole
OOD test set, no subsampling, one deterministic number per arm — the paper's protocol
(arXiv 2205.00403 §6.2.1, appendix C.1). Dispersion belongs across training seeds; see
scripts/metrics/cifar100_overnight_report.py.

The per-method directories below hold one CSV per dataset. Stage them from the inference
output tree (`${EXPERIMENTS_HOME}/${PROJECT_NAME}/infer/cifar100_<arm>__<dataset>/`)
with scripts/metrics/stage_cifar100_ood.sh, then:

    uv run python -m src.paper_helpers.ood_metrics.cifar100

See docs/models/CIFAR100_BENCHMARK.md.
"""
from pathlib import Path

from src.paper_helpers.ood_metrics.runner import run_ood_comparison

CIFAR_CSV_FILENAMES = {
    "cifar100": "cifar100.csv",
    "cifar10": "cifar10.csv",
    "svhn": "svhn.csv",
}

STAGING = Path("csv/ood_metrics")

SCORE_MODES = ("msp", "dempster_shafer")

# `cifar100last_*` is epoch 249, where the arms are actually compared; `cifar100_*` is
# best.ckpt, kept for the reference-only table. The literal arm has no last.ckpt.
CHECKPOINT_SETS = {
    "cifar100last": {
        "Baseline": "cifar100last_baseline",
        "SNGP": "cifar100last_sngp",
        "SpecReg (matched)": "cifar100last_specreg",
    },
    "cifar100": {
        "Baseline": "cifar100_baseline",
        "SNGP": "cifar100_sngp",
        "SpecReg (matched)": "cifar100_specreg",
        "SpecReg (literal)": "cifar100_specreg_literal",
    },
}

if __name__ == "__main__":
    for prefix, methods in CHECKPOINT_SETS.items():
        for score_mode in SCORE_MODES:
            run_ood_comparison(
                dataset="cifar100",
                methods={name: STAGING / d for name, d in methods.items()},
                # CIFAR-10 is the harder, near-OOD pair (same 32px natural-image
                # domain); SVHN is the far-OOD pair the SNGP paper headlines.
                ood_datasets=["cifar10", "svhn"],
                out_dir=STAGING,
                score_mode=score_mode,
                csv_filenames=CIFAR_CSV_FILENAMES,
                out_filename=f"{prefix}_ood_auroc_{score_mode}.csv",
                estimator="full_population",
            )
