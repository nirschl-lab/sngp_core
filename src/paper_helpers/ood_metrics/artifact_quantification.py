from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.metrics import (
	auc,
	average_precision_score,
	precision_recall_curve,
	roc_auc_score,
	roc_curve,
)

from src.metrics.io import normalized_entropy as _normalized_entropy_from_probs
from src.metrics.io import parse_float_list as _parse_class_probs


REAL_STREAM_CANDIDATES = {"real", "clean", "original", "id", "in", "in_distribution"}
ARTIFACT_STREAM_CANDIDATES = {
	"artifact",
	"artifacts",
	"simulated",
	"ood",
	"out",
	"out_distribution",
}


@dataclass
class DetectionMetrics:
	auroc_entropy: float
	auroc_confidence_score: float
	aupr_entropy: float
	aupr_confidence_score: float
	fpr95_entropy: float
	fpr95_confidence_score: float


ConfidenceScoreMode = Literal["raw_confidence", "uncertainty"]


def _build_confidence_score(
	df: pd.DataFrame,
	mode: ConfidenceScoreMode,
) -> tuple[np.ndarray, np.ndarray, str]:
	"""Return y_true, confidence-derived score, and a human-readable label.

	For `raw_confidence`, larger scores should indicate real samples, so positive class is real.
	For `uncertainty`, larger scores should indicate artifact samples, so positive class is artifact.
	"""
	confidence = df["confidence"].to_numpy()
	if mode == "raw_confidence":
		y_true = (df["stream_canonical"] == "real").astype(int).to_numpy()
		score = confidence
		score_label = "Confidence (real-positive)"
	elif mode == "uncertainty":
		y_true = (df["stream_canonical"] == "artifact").astype(int).to_numpy()
		score = 1.0 - confidence
		score_label = "1-Confidence (artifact-positive)"
	else:
		raise ValueError(f"Unsupported confidence score mode: {mode}")

	return y_true, score, score_label


def _canonicalize_stream(value: str) -> str:
	x = str(value).strip().lower()
	if x in REAL_STREAM_CANDIDATES:
		return "real"
	if x in ARTIFACT_STREAM_CANDIDATES:
		return "artifact"
	return x


def _resolve_stream_labels(stream_values: list[str]) -> tuple[str, str]:
	canon = [_canonicalize_stream(v) for v in stream_values]
	unique = sorted(set(canon))
	if "real" in unique and "artifact" in unique:
		return "real", "artifact"
	if len(unique) != 2:
		raise ValueError(
			"Expected exactly two stream groups (real and artifact). "
			f"Found: {unique}."
		)
	return unique[0], unique[1]


def _fpr_at_tpr95(y_true: np.ndarray, y_score: np.ndarray) -> float:
	fpr, tpr, _ = roc_curve(y_true, y_score)
	idx = np.where(tpr >= 0.95)[0]
	if idx.size == 0:
		return 1.0
	return float(np.min(fpr[idx]))


def _build_stream_df(df: pd.DataFrame) -> pd.DataFrame:
	required_cols = [
		"image_id",
		"target",
		"prediction",
		"confidence",
		"class_probs",
		"stream",
	]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise KeyError(f"Missing required columns: {missing}")

	out = df.copy()
	out["class_probs_parsed"] = out["class_probs"].map(_parse_class_probs)
	out = out[out["class_probs_parsed"].notna()].copy()
	out["entropy"] = out["class_probs_parsed"].map(_normalized_entropy_from_probs)
	out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
	out["target"] = pd.to_numeric(out["target"], errors="coerce")
	out["prediction"] = pd.to_numeric(out["prediction"], errors="coerce")
	out = out.dropna(subset=["confidence", "target", "prediction"])
	out["correct"] = (out["target"].astype(int) == out["prediction"].astype(int)).astype(int)
	out["stream_canonical"] = out["stream"].map(_canonicalize_stream)
	return out


def _compute_detection_metrics(
	df: pd.DataFrame,
	real_label: str,
	artifact_label: str,
	confidence_score_mode: ConfidenceScoreMode,
) -> DetectionMetrics:
	y_true = (df["stream_canonical"] == artifact_label).astype(int).to_numpy()
	entropy_score = df["entropy"].to_numpy()
	y_true_conf, confidence_score, _ = _build_confidence_score(df, confidence_score_mode)

	fpr_e, tpr_e, _ = roc_curve(y_true, entropy_score)
	fpr_c, tpr_c, _ = roc_curve(y_true_conf, confidence_score)
	_ = (fpr_e, tpr_e, fpr_c, tpr_c)

	return DetectionMetrics(
		auroc_entropy=float(roc_auc_score(y_true, entropy_score)),
		auroc_confidence_score=float(roc_auc_score(y_true_conf, confidence_score)),
		aupr_entropy=float(average_precision_score(y_true, entropy_score)),
		aupr_confidence_score=float(average_precision_score(y_true_conf, confidence_score)),
		fpr95_entropy=_fpr_at_tpr95(y_true, entropy_score),
		fpr95_confidence_score=_fpr_at_tpr95(y_true_conf, confidence_score),
	)


def _compute_paired_metrics(df: pd.DataFrame, real_label: str, artifact_label: str) -> dict[str, float]:
	base_cols = ["image_id", "target", "prediction", "confidence", "entropy", "correct"]
	real_df = df[df["stream_canonical"] == real_label][base_cols].rename(
		columns={
			"prediction": "prediction_real",
			"confidence": "confidence_real",
			"entropy": "entropy_real",
			"correct": "correct_real",
		}
	)
	artifact_df = df[df["stream_canonical"] == artifact_label][base_cols].rename(
		columns={
			"prediction": "prediction_artifact",
			"confidence": "confidence_artifact",
			"entropy": "entropy_artifact",
			"correct": "correct_artifact",
		}
	)

	merged = real_df.merge(artifact_df, on=["image_id", "target"], how="inner")
	if merged.empty:
		raise ValueError("No paired rows found between real and artifact streams by image_id+target.")

	merged["delta_confidence"] = merged["confidence_artifact"] - merged["confidence_real"]
	merged["delta_entropy"] = merged["entropy_artifact"] - merged["entropy_real"]
	merged["flip_pred"] = (merged["prediction_real"] != merged["prediction_artifact"]).astype(int)

	n = float(len(merged))
	return {
		"paired_count": n,
		"acc_real": float(np.mean(merged["correct_real"])),
		"acc_artifact": float(np.mean(merged["correct_artifact"])),
		"acc_drop": float(np.mean(merged["correct_real"]) - np.mean(merged["correct_artifact"])),
		"mean_confidence_real": float(np.mean(merged["confidence_real"])),
		"mean_confidence_artifact": float(np.mean(merged["confidence_artifact"])),
		"confidence_drop": float(np.mean(merged["confidence_real"]) - np.mean(merged["confidence_artifact"])),
		"mean_entropy_real": float(np.mean(merged["entropy_real"])),
		"mean_entropy_artifact": float(np.mean(merged["entropy_artifact"])),
		"entropy_rise": float(np.mean(merged["entropy_artifact"]) - np.mean(merged["entropy_real"])),
		"prediction_flip_rate": float(np.mean(merged["flip_pred"])),
	}


def _plot_roc_curves(
	model_name: str,
	df: pd.DataFrame,
	save_path: Path,
	confidence_score_mode: ConfidenceScoreMode,
) -> None:
	y_true = (df["stream_canonical"] == "artifact").astype(int).to_numpy()
	entropy_score = df["entropy"].to_numpy()
	y_true_conf, confidence_score, confidence_label = _build_confidence_score(df, confidence_score_mode)

	fpr_e, tpr_e, _ = roc_curve(y_true, entropy_score)
	fpr_m, tpr_m, _ = roc_curve(y_true_conf, confidence_score)
	auc_e = auc(fpr_e, tpr_e)
	auc_m = auc(fpr_m, tpr_m)

	plt.figure(figsize=(7, 6))
	plt.plot(fpr_e, tpr_e, lw=2, label=f"Entropy (AUROC={auc_e:.4f})")
	plt.plot(fpr_m, tpr_m, lw=2, label=f"{confidence_label} (AUROC={auc_m:.4f})")
	plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
	plt.xlim(0, 1)
	plt.ylim(0, 1.01)
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title(f"Artifact Detection ROC - {model_name}")
	plt.legend(frameon=False)
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig(save_path, dpi=250)
	plt.close()


def _kde_reflection_boundary_corrected(
	values: np.ndarray,
	x: np.ndarray,
	lower_bound: float,
	upper_bound: float,
	bw_method: str | float = "scott",
) -> np.ndarray:
	"""Evaluate KDE with simple reflection at both bounds to reduce edge bias."""
	reflected_low = 2.0 * lower_bound - values
	reflected_high = 2.0 * upper_bound - values
	augmented = np.concatenate([values, reflected_low, reflected_high])
	kde = gaussian_kde(augmented, bw_method=bw_method)
	y = kde(x) * (1.0 / 3.0)
	return y


def _plot_uncertainty_kde(model_name: str, df: pd.DataFrame, save_path: Path) -> None:
	real = df.loc[df["stream_canonical"] == "real", "entropy"].to_numpy()
	artifact = df.loc[df["stream_canonical"] == "artifact", "entropy"].to_numpy()

	if real.size == 0 or artifact.size == 0:
		return

	lower_bound = min(0.0, float(real.min()), float(artifact.min()))
	upper_bound = max(1.0, float(real.max()), float(artifact.max()))
	if np.isclose(lower_bound, upper_bound):
		lower_bound -= 0.1
		upper_bound += 0.1

	span = upper_bound - lower_bound
	pad = 0.08 * span
	x_min = lower_bound - pad
	x_max = upper_bound + pad

	x = np.linspace(x_min, x_max, 512)
	plt.figure(figsize=(7, 6))

	for values, label, style in [
		(real, "Real", "-"),
		(artifact, "Artifact", "--"),
	]:
		if np.unique(values).size < 2 or np.isclose(np.var(values), 0.0):
			plt.axvline(float(values[0]), linestyle=style, lw=2, alpha=0.8, label=f"{label} (constant)")
			continue
		y = _kde_reflection_boundary_corrected(
			values=values,
			x=x,
			lower_bound=lower_bound,
			upper_bound=upper_bound,
			bw_method="scott",
		)
		plt.plot(x, y, linestyle=style, lw=2, label=label)
		plt.fill_between(x, y, 0, alpha=0.15)

	plt.xlabel("Normalized entropy")
	plt.ylabel("Density")
	plt.title(f"Uncertainty Shift (Entropy) - {model_name}")
	plt.legend(frameon=False)
	plt.grid(alpha=0.25)
	plt.tight_layout()
	plt.savefig(save_path, dpi=250)
	plt.close()


def _plot_combined_uncertainty_summary(
	entropy_by_model: dict[str, dict[str, np.ndarray]],
	save_path: Path,
) -> None:
	"""Plot all model entropy curves in one figure for direct comparison.

	The figure uses two panels:
	- Left: real stream entropy KDE for all models.
	- Right: artifact stream entropy KDE for all models.
	"""
	if not entropy_by_model:
		return

	all_real = [v["real"] for v in entropy_by_model.values() if v["real"].size > 0]
	all_artifact = [v["artifact"] for v in entropy_by_model.values() if v["artifact"].size > 0]
	if not all_real or not all_artifact:
		return

	global_min = min(float(np.min(arr)) for arr in (all_real + all_artifact))
	global_max = max(float(np.max(arr)) for arr in (all_real + all_artifact))
	lower_bound = min(0.0, global_min)
	upper_bound = max(1.0, global_max)
	if np.isclose(lower_bound, upper_bound):
		lower_bound -= 0.1
		upper_bound += 0.1

	span = upper_bound - lower_bound
	pad = 0.08 * span
	x = np.linspace(lower_bound - pad, upper_bound + pad, 512)

	fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
	colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(entropy_by_model))))

	for (model_name, streams), color in zip(entropy_by_model.items(), colors):
		for ax, stream_name, title in [
			(axes[0], "real", "Real Stream Entropy"),
			(axes[1], "artifact", "Artifact Stream Entropy"),
		]:
			values = streams[stream_name]
			if values.size == 0:
				continue

			if values.size < 2 or np.isclose(np.var(values), 0.0):
				ax.axvline(
					float(values[0]),
					color=color,
					linestyle="-",
					lw=2,
					alpha=0.85,
					label=f"{model_name} (constant)",
				)
				ax.set_title(title)
				continue

			y = _kde_reflection_boundary_corrected(
				values=values,
				x=x,
				lower_bound=lower_bound,
				upper_bound=upper_bound,
			)
			ax.plot(x, y, color=color, lw=2, label=model_name)
			ax.fill_between(x, y, 0, color=color, alpha=0.1)
			ax.set_title(title)

	for ax in axes:
		ax.set_xlabel("Normalized entropy")
		ax.grid(alpha=0.25)

	axes[0].set_ylabel("Density")
	axes[0].legend(frameon=False, fontsize=9)
	axes[1].legend(frameon=False, fontsize=9)
	fig.suptitle("Combined Entropy Comparison Across Models")
	fig.tight_layout()
	fig.savefig(save_path, dpi=250)
	plt.close(fig)


def _plot_comparison_bars(results_df: pd.DataFrame, save_path: Path) -> None:
	fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

	axes[0].bar(results_df["model"], results_df["acc_drop"], color="#d95f02")
	axes[0].set_title("Accuracy Drop")
	axes[0].set_ylabel("real - artifact")
	axes[0].tick_params(axis="x", rotation=25)

	axes[1].bar(results_df["model"], results_df["entropy_rise"], color="#1b9e77")
	axes[1].set_title("Entropy Rise")
	axes[1].set_ylabel("artifact - real")
	axes[1].tick_params(axis="x", rotation=25)

	axes[2].bar(results_df["model"], results_df["auroc_entropy"], color="#7570b3")
	axes[2].set_title("Artifact Detection AUROC")
	axes[2].set_ylabel("Using entropy")
	axes[2].set_ylim(0, 1)
	axes[2].tick_params(axis="x", rotation=25)

	for ax in axes:
		ax.grid(axis="y", alpha=0.25)

	fig.tight_layout()
	fig.savefig(save_path, dpi=250)
	plt.close(fig)


def _load_paired_csv(csv_path: str | Path, real_csv_path: str | Path | None) -> pd.DataFrame:
	"""Read one model's artifact predictions, joining in a shared real-stream CSV if needed.

	A sweep run written with `infer.save.streams=[artifact]` has no "real" rows of its own
	by design -- the real/clean stream is bit-identical for every count/severity variant
	against the same checkpoint, so it's computed once into a shared `real_baseline/
	predictions.csv` instead of being recomputed and rewritten per sweep run (see
	docs/DATASETS.md's "Saving sweep results" section). `_build_stream_df` (and everything
	downstream of it here) still expects one dataframe with both streams, so concatenate
	them before that call rather than touching the pairing logic itself.
	"""
	df_raw = pd.read_csv(csv_path)
	has_real = "stream" in df_raw.columns and (df_raw["stream"].map(_canonicalize_stream) == "real").any()
	if has_real or real_csv_path is None:
		return df_raw
	real_df_raw = pd.read_csv(real_csv_path)
	return pd.concat([real_df_raw, df_raw], ignore_index=True)


def quantify_artifact_impact(
	model_csv_map: dict[str, str | Path],
	output_dir: str | Path,
	confidence_score_mode: ConfidenceScoreMode = "raw_confidence",
	real_csv_map: dict[str, str | Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""`real_csv_map` (optional): `{model_name: real_baseline_csv_path}` for models whose
	`model_csv_map` entry holds only the artifact stream (a count/severity sweep run) --
	its rows are joined in before pairing. A model already holding both streams in one CSV
	(the pre-sweep, self-contained format) needs no entry here.
	"""
	out_dir = Path(output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	(out_dir / "plots").mkdir(parents=True, exist_ok=True)

	summary_rows: list[dict[str, Any]] = []
	pr_rows: list[dict[str, Any]] = []
	entropy_by_model: dict[str, dict[str, np.ndarray]] = {}

	for model_name, csv_path in model_csv_map.items():
		df_raw = _load_paired_csv(csv_path, (real_csv_map or {}).get(model_name))
		df = _build_stream_df(df_raw)

		real_label, artifact_label = _resolve_stream_labels(df["stream_canonical"].tolist())
		# Normalize to expected names for simpler plotting/aggregation.
		df.loc[df["stream_canonical"] == real_label, "stream_canonical"] = "real"
		df.loc[df["stream_canonical"] == artifact_label, "stream_canonical"] = "artifact"
		entropy_by_model[model_name] = {
			"real": df.loc[df["stream_canonical"] == "real", "entropy"].to_numpy(),
			"artifact": df.loc[df["stream_canonical"] == "artifact", "entropy"].to_numpy(),
		}

		det = _compute_detection_metrics(
			df,
			"real",
			"artifact",
			confidence_score_mode=confidence_score_mode,
		)
		paired = _compute_paired_metrics(df, "real", "artifact")

		row = {
			"model": model_name,
			"confidence_score_mode": confidence_score_mode,
			"n_rows": int(len(df)),
			**paired,
			"auroc_entropy": det.auroc_entropy,
			"auroc_confidence_score": det.auroc_confidence_score,
			"aupr_entropy": det.aupr_entropy,
			"aupr_confidence_score": det.aupr_confidence_score,
			"fpr95_entropy": det.fpr95_entropy,
			"fpr95_confidence_score": det.fpr95_confidence_score,
		}
		summary_rows.append(row)

		y_true = (df["stream_canonical"] == "artifact").astype(int).to_numpy()
		pr_entropy_prec, pr_entropy_rec, _ = precision_recall_curve(y_true, df["entropy"].to_numpy())
		y_true_conf, confidence_score, _ = _build_confidence_score(df, confidence_score_mode)
		pr_msp_prec, pr_msp_rec, _ = precision_recall_curve(y_true_conf, confidence_score)
		pr_rows.append(
			{
				"model": model_name,
				"confidence_score_mode": confidence_score_mode,
				"pr_auc_entropy_check": float(auc(pr_entropy_rec, pr_entropy_prec)),
				"pr_auc_confidence_score_check": float(auc(pr_msp_rec, pr_msp_prec)),
			}
		)

		_plot_roc_curves(
			model_name=model_name,
			df=df,
			save_path=out_dir / "plots" / f"{model_name}_roc.png",
			confidence_score_mode=confidence_score_mode,
		)
		_plot_uncertainty_kde(
			model_name=model_name,
			df=df,
			save_path=out_dir / "plots" / f"{model_name}_entropy_kde.png",
		)

	summary_df = pd.DataFrame(summary_rows).sort_values(
		by=["acc_drop", "auroc_entropy"], ascending=[True, False]
	)
	pr_df = pd.DataFrame(pr_rows)

	summary_csv = out_dir / "artifact_quantification_summary.csv"
	summary_df.to_csv(summary_csv, index=False)
	pr_df.to_csv(out_dir / "artifact_quantification_pr_checks.csv", index=False)

	_plot_comparison_bars(summary_df, save_path=out_dir / "plots" / "model_comparison_bars.png")
	_plot_combined_uncertainty_summary(
		entropy_by_model=entropy_by_model,
		save_path=out_dir / "plots" / "combined_entropy_all_models.png",
	)

	print(f"Saved summary metrics to: {summary_csv}")
	print(f"Saved plots to: {out_dir / 'plots'}")
	return summary_df, pr_df


if __name__ == "__main__":
    # Replace with your own model->csv mappings.
    base_path = Path('/data1/maheswararao/experiments/sngp/infer/')

    # MODEL_CSV_MAP = {
    #     "baseline": Path(base_path, 'acevedo_baseline', 'predictions.csv'),
    #     "mc_dropout": Path(base_path, 'acevedo_mc', 'predictions.csv'),
    #     "sngp": Path(base_path, 'acevedo_sngp', 'predictions.csv'),
    # }
    # OUTPUT_DIR = Path(base_path, 'acevedo_artifact_quantification')

    MODEL_CSV_MAP = {
            "baseline": Path(base_path, 'wong_baseline', 'predictions.csv'),
            "mc_dropout": Path(base_path, 'wong_mc', 'predictions.csv'),
            "sngp": Path(base_path, 'wong_sngp', 'predictions.csv'),
        }
    OUTPUT_DIR = Path(base_path, 'wong_artifact_quantification')
        

    CONFIDENCE_SCORE_MODE: ConfidenceScoreMode = "raw_confidence"

    if not MODEL_CSV_MAP:
        raise ValueError("Please fill MODEL_CSV_MAP with {model_name: csv_path} entries.")

    quantify_artifact_impact(
        model_csv_map=MODEL_CSV_MAP,
        output_dir=OUTPUT_DIR,
        confidence_score_mode=CONFIDENCE_SCORE_MODE,
    )
