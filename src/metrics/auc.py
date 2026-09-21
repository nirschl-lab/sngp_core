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
        if "prediction_prob_score" not in df.columns:
            raise KeyError("Missing required column 'prediction_prob_score' for score_mode='msp'")
        return pd.to_numeric(df["prediction_prob_score"], errors="coerce").dropna()

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
    """`fold_policy` defaults to `LEGACY_ISBI_FOLD_POLICY`: filters the ID frame to
    `fold=='test'` but leaves the OOD frame unfiltered. That asymmetry is not a bug --
    it's load-bearing for every already-published number in csv/final/ and
    csv/isbi_test_files/ -- so it stays the default; pass `SYMMETRIC_FOLD_POLICY`
    (or a custom `OodFoldPolicy`) to filter both frames the same way instead.
    """
    res = {}
    for id_name in id_names:
        for ood_name in ood_names:
            id_df = pd.read_csv(os.path.join(csv_path, csv_file_names[id_name]))
            if fold_policy.id_fold is not None:
                id_df = id_df[id_df["fold"] == fold_policy.id_fold]
            ood_df = pd.read_csv(os.path.join(csv_path, csv_file_names[ood_name]))
            if fold_policy.ood_fold is not None:
                ood_df = ood_df[ood_df["fold"] == fold_policy.ood_fold]
            id_scores = _compute_ood_score_series(id_df, score_mode)
            ood_scores = _compute_ood_score_series(ood_df, score_mode)

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