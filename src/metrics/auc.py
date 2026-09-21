from typing import Dict, Sequence

from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import os

from src.metrics.dempster_shafer_uncertainity import DempsterShaferUncertainty
from src.metrics.io import (
    LEGACY_ISBI_FOLD_POLICY,
    SYMMETRIC_FOLD_POLICY,  # noqa: F401 -- re-exported for callers of this module
    OodFoldPolicy,
    normalized_entropy,
    parse_float_list,
)

# Score modes whose value rises with uncertainty (so AUROC must not invert them).
DEMPSTER_SHAFER_SCORE_MODES = frozenset({"dempster_shafer", "ds"})
UNCERTAINTY_SCORE_MODES = frozenset({"entropy", "uncertainty"}) | DEMPSTER_SHAFER_SCORE_MODES

# Where max-softmax-probability lives, in precedence order. The two names are the same
# quantity under the project's two CSV schemas: `prediction_prob_score` is written by
# src/callbacks/test_artifacts_callback.py, `confidence` by src/inference/records.py
# (`outputs.probs.max(dim=1).values`). See .claude/skills/metrics/references/csv_schema.md.
#
# Precedence, not preference: the callback name is checked first so every already-published
# call site resolves to exactly the column it resolved to before this fallback existed.
# `src.metrics.io.load_predictions` refuses frames carrying both, so on a real CSV the
# order can never silently pick the wrong one -- but that guarantee lives in another
# module, so it is pinned by a test here too.
MSP_SCORE_COLUMNS = ("prediction_prob_score", "confidence")

seeds = [42, 1337, 12345, 8675309, 314159, 271828, 20240427, 987654321, 3735928559, 777]
sample_rate = 1000

# Thin aliases: the canonical implementations now live in src/metrics/io.py, shared
# with src.metrics.artifact_quantification (which had its own byte-identical copies)
# and the new PredictionFrame loader. Kept under these names so existing importers of
# this module (calculate_ood_metrics.py, predictive_entropy.py) see no behavior change.
_parse_class_probs = parse_float_list
_normalized_entropy = normalized_entropy


def _compute_ood_score_series(df, score_mode):
    if score_mode == "msp":
        column = next((c for c in MSP_SCORE_COLUMNS if c in df.columns), None)
        if column is None:
            raise KeyError(
                "Missing required column for score_mode='msp': need either "
                "'prediction_prob_score' (callback schema) or 'confidence' "
                "(inference schema)."
            )
        return pd.to_numeric(df[column], errors="coerce").dropna()

    if score_mode in {"entropy", "uncertainty"}:
        if "class_probs" not in df.columns:
            raise KeyError("Missing required column 'class_probs' for score_mode='entropy'")
        probs = df["class_probs"].map(_parse_class_probs)
        entropy = probs.map(lambda x: _normalized_entropy(x) if x is not None else np.nan)
        return entropy.dropna()

    if score_mode in DEMPSTER_SHAFER_SCORE_MODES:
        # `K / (K + sum_c exp(logit_c))` -- total evidence mass, not distribution shape.
        # This is the score the SNGP reference itself uses for CIFAR OOD detection
        # (`dempster_shafer_ood` in baselines/cifar/sngp.py), and the one worth reaching
        # for with a GP head: softmax is shift-invariant, so MSP and entropy throw away
        # exactly the logit magnitude SNGP is trained to modulate.
        #
        # Prefer the persisted column: `src/inference/infer.py` writes it from
        # `src.metrics.uncertainty.dempster_shafer`, which reproduces
        # `DempsterShaferUncertainty` in float64. Older CSVs predate the column but do
        # carry `class_logits`, so fall back to recomputing rather than failing.
        if "dempster_shafer" in df.columns:
            series = pd.to_numeric(df["dempster_shafer"], errors="coerce").dropna()
            if not series.empty:
                return series
        if "class_logits" not in df.columns:
            raise KeyError(
                "Need either a 'dempster_shafer' column or 'class_logits' for "
                "score_mode='dempster_shafer'. Note probabilities are not enough: "
                "softmax discards the total evidence this score measures."
            )
        logits = df["class_logits"].map(_parse_class_probs)
        valid = logits[logits.map(lambda x: x is not None and len(x) > 0)]
        if valid.empty:
            return pd.Series(dtype=float)
        scores = DempsterShaferUncertainty(np.vstack(valid.to_numpy()))
        return pd.Series(scores, index=valid.index).dropna()

    raise ValueError(
        "Unsupported score_mode. Use 'msp', 'entropy', or 'dempster_shafer'."
    )


def _read_ood_score_series(csv_path, csv_file_names, name, score_mode, fold):
    """Read one dataset's prediction CSV, optionally fold-filter it, and reduce it to a
    score series.

    Shared by both estimators below so neither re-implements the read. Keeps the inline
    `df[df["fold"] == fold]` rather than calling `io.filter_fold`, which raises earlier
    and with different exception types on an absent/empty fold -- this helper is on the
    frozen path, so its failure modes must not move.
    """
    df = pd.read_csv(os.path.join(csv_path, csv_file_names[name]))
    if fold is not None:
        df = df[df["fold"] == fold]
    # Module-global lookup on purpose, so tests can monkeypatch it -- do not rebind this
    # to a local or a `from ... import` name.
    return _compute_ood_score_series(df, score_mode)


def AUROC(ID_MSP, OOD_MSP, score_is_uncertainty=False):
    # Convert to numpy arrays
    ID_MSP = np.asarray(ID_MSP)
    OOD_MSP = np.asarray(OOD_MSP)
    
    # Build uncertainty scores where larger values indicate OOD.
    if score_is_uncertainty:
        id_uncertainty = ID_MSP
        ood_uncertainty = OOD_MSP
    else:
        id_uncertainty = 1 - ID_MSP # should be 0 in ideal case
        ood_uncertainty = 1 - OOD_MSP # should be 1 in ideal case
    
    # Concatenate scores and labels
    y_true = np.concatenate([np.zeros_like(ID_MSP), np.ones_like(OOD_MSP)])  # 0 = ID, 1 = OOD
    y_scores = np.concatenate([id_uncertainty, ood_uncertainty])
    
    # y_true = np.concatenate([np.ones_like(ID_MSP), np.zeros_like(OOD_MSP)])  # 0 = ID, 1 = OOD
    # y_scores = np.concatenate([ID_MSP, OOD_MSP])

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)
    return auroc

# def AUROC_across_dataset(csv_path, csv_file_names, id_names, ood_names):
#     for id_name in id_names:
#         for ood_name in ood_names:
#             id_df = pd.read_csv(os.path.join(csv_path, csv_file_names[id_name]))
#             id_df = id_df[id_df['fold']=='test']
#             ood_df = pd.read_csv(os.path.join(csv_path, csv_file_names[ood_name]))
#             #calcualte mean and std across 10 different seeds
#             auroc_list = []
#             for seed in seeds:
#                 id_samples = id_df.sample(sample_rate, random_state=seed)['prediction_prob_score']
#                 ood_samples = ood_df.sample(sample_rate, random_state=seed)['prediction_prob_score']
#                 auroc_list.append(AUROC(id_samples, ood_samples))
#             print(f"AUROC {id_name} vs {ood_name} - {np.mean(auroc_list):.4f} ± {np.std(auroc_list):.4f}")


def AUROC_across_dataset(
    csv_path,
    csv_file_names,
    id_names,
    ood_names,
    score_mode="msp",
    fold_policy: OodFoldPolicy = LEGACY_ISBI_FOLD_POLICY,
):
    """Mean +- std over 10 fixed-seed subsamples of <=`sample_rate` rows per frame.

    Frozen: this reproduces every published number in csv/final/ and
    csv/isbi_test_files/. Use it for those. For anything reporting the SNGP paper's
    protocol, use `AUROC_across_dataset_full_population` instead -- the two will not
    agree to 4 decimals and are not meant to.

    Note the `+-` here is *not* a bootstrap: `Series.sample` defaults to
    `replace=False`, so this is subsampling without replacement.

    `fold_policy` defaults to `LEGACY_ISBI_FOLD_POLICY`: filters the ID frame to
    `fold=='test'` but leaves the OOD frame unfiltered. That asymmetry is not a bug --
    it's load-bearing for every already-published number in csv/final/ and
    csv/isbi_test_files/ -- so it stays the default; pass `SYMMETRIC_FOLD_POLICY`
    (or a custom `OodFoldPolicy`) to filter both frames the same way instead.
    """
    res = {}
    for id_name in id_names:
        for ood_name in ood_names:
            id_scores = _read_ood_score_series(
                csv_path, csv_file_names, id_name, score_mode, fold_policy.id_fold
            )
            ood_scores = _read_ood_score_series(
                csv_path, csv_file_names, ood_name, score_mode, fold_policy.ood_fold
            )

            n_samples = min(sample_rate, len(id_scores), len(ood_scores))
            if n_samples == 0:
                raise ValueError(
                    f"No valid rows available for AUROC with score_mode='{score_mode}' "
                    f"for ID='{id_name}' and OOD='{ood_name}'."
                )

            score_is_uncertainty = score_mode in UNCERTAINTY_SCORE_MODES
            #calcualte mean and std across 10 different seeds
            auroc_list = []
            for seed in seeds:
                id_samples = id_scores.sample(n_samples, random_state=seed)
                ood_samples = ood_scores.sample(n_samples, random_state=seed)
                auroc_list.append(
                    AUROC(id_samples, ood_samples, score_is_uncertainty=score_is_uncertainty)
                )
            # res[ood_name] = (np.mean(auroc_list), np.std(auroc_list))
            res[ood_name] = f"{np.mean(auroc_list):.4f} ± {np.std(auroc_list):.4f}"
        # res[id_name] = sub_res
    return res


def AUROC_across_dataset_full_population(
    csv_path: str,
    csv_file_names: Dict[str, str],
    id_name: str,
    ood_names: Sequence[str],
    *,
    score_mode: str = "msp",
    fold_policy: OodFoldPolicy = SYMMETRIC_FOLD_POLICY,
) -> Dict[str, float]:
    """One deterministic AUROC per OOD dataset, over every row of both test sets.

    The protocol of the SNGP paper (Liu et al., arXiv 2205.00403, section 6.2.1 /
    appendix C.1): the full in-distribution test set against the full OOD test set, no
    subsampling, so the number is a property of the model rather than of a resampling
    seed. Dispersion, where reported, belongs across *training* seeds -- this returns one
    float per OOD dataset and takes no position on how those floats are aggregated.

    No new estimator is needed for this: `AUROC` already computes the full-population
    number, since it contains no sampling of its own.

    Unequal group sizes are fine and deliberate -- CIFAR-100's 10,000 test rows against
    SVHN's 26,032. AUROC is a rank statistic estimating `P(score_OOD > score_ID)`, which
    is invariant to the ID:OOD ratio; the sibling function's `min(...)` balancing was an
    artifact of subsampling, not a requirement.

    The deliberate counterpart to `AUROC_across_dataset`, which returns `"mean +- std"`
    strings over 10 fixed-seed subsamples of <=1000 rows. That one is frozen for
    published ISBI numbers; this one is for the SNGP protocol. Three differences beyond
    the estimator, all intentional:

      - `id_name` is scalar, not a list. The list form keys its result dict by OOD name
        alone, so a second ID dataset silently overwrote the first's results.
      - `fold_policy` defaults to `SYMMETRIC_FOLD_POLICY`: test-vs-test is what the
        protocol says, and no published number depends on this function, so the legacy
        asymmetry has nothing to be load-bearing for here.
      - The ID frame is read and scored once, not once per OOD dataset.
    """
    id_scores = _read_ood_score_series(
        csv_path, csv_file_names, id_name, score_mode, fold_policy.id_fold
    )
    score_is_uncertainty = score_mode in UNCERTAINTY_SCORE_MODES

    res: Dict[str, float] = {}
    for ood_name in ood_names:
        ood_scores = _read_ood_score_series(
            csv_path, csv_file_names, ood_name, score_mode, fold_policy.ood_fold
        )
        if len(id_scores) == 0 or len(ood_scores) == 0:
            raise ValueError(
                f"No valid rows available for AUROC with score_mode='{score_mode}' for "
                f"ID='{id_name}' ({len(id_scores)} rows) and OOD='{ood_name}' "
                f"({len(ood_scores)} rows)."
            )
        res[ood_name] = float(
            AUROC(id_scores, ood_scores, score_is_uncertainty=score_is_uncertainty)
        )
    return res