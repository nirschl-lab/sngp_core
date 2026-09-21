"""CIFAR-100 (in-distribution) vs CIFAR-10 / SVHN (OOD) AUROC across the four
WideResNet-28-10 benchmark arms. Config-only call site; the comparison loop itself lives
in src/paper_helpers/ood_metrics/runner.py.

Scored with **Dempster-Shafer uncertainty**, not max-softmax-probability. Softmax is
shift-invariant, so MSP and entropy discard the total logit magnitude — which is exactly
what an SNGP head is trained to modulate, and therefore exactly the signal this
comparison is about. It is also the score the reference uses for CIFAR OOD detection
(`dempster_shafer_ood` in google/uncertainty-baselines `baselines/cifar/sngp.py`).

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

if __name__ == "__main__":
    run_ood_comparison(
        dataset="cifar100",
        methods={
            "Baseline": STAGING / "cifar100_baseline",
            "SNGP": STAGING / "cifar100_sngp",
            "SpecReg (matched)": STAGING / "cifar100_specreg",
            "SpecReg (literal)": STAGING / "cifar100_specreg_literal",
        },
        # CIFAR-10 is the harder, near-OOD pair (same 32px natural-image domain);
        # SVHN is the far-OOD pair the SNGP paper headlines.
        ood_datasets=["cifar10", "svhn"],
        out_dir=STAGING,
        score_mode="dempster_shafer",
        csv_filenames=CIFAR_CSV_FILENAMES,
        out_filename="cifar100_ood_auroc_dempster_shafer.csv",
    )
