from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics.classification import (
	Accuracy,
	MulticlassCalibrationError,
	MulticlassF1Score,
	MulticlassPrecision,
	MulticlassRecall,
)


def _resolve_device(device_name: str) -> torch.device:
	if device_name == "cuda" and not torch.cuda.is_available():
		logger.warning("CUDA requested but unavailable. Falling back to CPU.")
		return torch.device("cpu")
	return torch.device(device_name)


def _to_cpu_tensor(x: Any) -> Optional[torch.Tensor]:
	if x is None:
		return None
	if torch.is_tensor(x):
		return x.detach().cpu()
	return torch.as_tensor(x).detach().cpu()


def _save_tensor_image(tensor: torch.Tensor, image_path: Path) -> None:
	from PIL import Image

	image = tensor.detach().cpu()
	if image.ndim == 4:
		image = image[0]
	if image.ndim == 3 and image.shape[0] in (1, 3):
		image = image.permute(1, 2, 0)

	array = image.numpy()
	array = np.nan_to_num(array)
	if array.min() < 0.0 or array.max() > 1.0:
		array = (array - array.min()) / (array.max() - array.min() + 1e-8)
	array = (array * 255.0).clip(0, 255).astype(np.uint8)

	if array.ndim == 3 and array.shape[-1] == 1:
		array = array.squeeze(-1)

	Image.fromarray(array).save(image_path)


def _extract_logits_probs(
	model: torch.nn.Module,
	x: torch.Tensor,
	use_mc_dropout: bool,
	mc_passes: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
	if use_mc_dropout and hasattr(model, "mc_predict"):
		result = model.mc_predict(x, T=mc_passes, return_std=True, apply_softmax=True)
		if isinstance(result, tuple) and len(result) == 3:
			logits, probs, uncertainty = result
			return logits, probs, uncertainty

	output = model(x)

	uncertainty = None
	if isinstance(output, tuple):
		logits = output[0]
		if len(output) >= 3 and torch.is_tensor(output[2]):
			uncertainty = output[2]
	else:
		logits = output

	probs = torch.softmax(logits, dim=1)
	return logits, probs, uncertainty


def _build_metrics(num_classes: int, metric_names: Sequence[str]) -> Dict[str, Any]:
	metric_names = set(metric_names)
	metrics: Dict[str, Any] = {}

	if "acc" in metric_names:
		metrics["acc"] = Accuracy(task="multiclass", num_classes=num_classes)
	if "ece" in metric_names:
		metrics["ece"] = MulticlassCalibrationError(num_classes=num_classes, n_bins=10, norm="l1")
	if "precision" in metric_names:
		metrics["precision"] = MulticlassPrecision(num_classes=num_classes, average="macro")
	if "recall" in metric_names:
		metrics["recall"] = MulticlassRecall(num_classes=num_classes, average="macro")
	if "f1" in metric_names:
		metrics["f1"] = MulticlassF1Score(num_classes=num_classes, average="macro")

	return metrics


class ArtifactInferenceRunner:
	"""Artifact-specific inference runner for paired real/artifact images."""

	def __init__(
		self,
		model: torch.nn.Module,
		dataloader: DataLoader,
		cfg: DictConfig,
		class_names: Optional[Dict[int, str]] = None,
		expected_num_classes: Optional[int] = None,
	) -> None:
		self.model = model
		self.dataloader = dataloader
		self.cfg = cfg
		self.class_names = class_names or {}
		self.expected_num_classes = expected_num_classes

		self.device = _resolve_device(str(self.cfg.infer.model.device))
		self.model.to(self.device)
		self.model.eval()

		save_root = Path(str(self.cfg.save_path))
		run_name = str(self.cfg.infer.save.run_name)
		self.output_root = save_root / run_name if run_name else save_root
		self.output_root.mkdir(parents=True, exist_ok=True)

		self._records: List[Dict[str, Any]] = []
		# per-stream metric states, e.g. {"real": {"acc": ..., ...}, "artifact": {...}}
		self._metric_states: Dict[str, Dict[str, Any]] = {}
		self._skip_metrics = False
		self._skip_metrics_reason = ""

	def _targets_match_expected_classes(self, targets: torch.Tensor) -> bool:
		if targets.numel() == 0:
			return True
		if self.expected_num_classes is None:
			return True
		min_target = int(targets.min().item())
		max_target = int(targets.max().item())
		return min_target >= 0 and max_target < self.expected_num_classes

	def _check_metric_compatibility(self, targets: torch.Tensor, output_num_classes: int) -> None:
		if self._skip_metrics or not bool(self.cfg.infer.metrics.enabled):
			return

		if self.expected_num_classes is not None and output_num_classes != self.expected_num_classes:
			self._skip_metrics = True
			self._metric_states = {}
			self._skip_metrics_reason = (
				"Skipping metrics because datamodule num_classes does not match model outputs "
				f"(datamodule num_classes={self.expected_num_classes}, model outputs={output_num_classes})."
			)
			return

		if not self._targets_match_expected_classes(targets):
			target_min = int(targets.min().item()) if targets.numel() > 0 else -1
			target_max = int(targets.max().item()) if targets.numel() > 0 else -1
			expected = self.expected_num_classes if self.expected_num_classes is not None else output_num_classes
			self._skip_metrics = True
			self._metric_states = {}
			self._skip_metrics_reason = (
				"Skipping metrics because targets are outside expected class range "
				f"(expected num_classes={expected}, observed target range=[{target_min}, {target_max}])."
			)

	def _ensure_metrics(self, num_classes: int, stream_name: str) -> None:
		if stream_name in self._metric_states:
			return
		stream_metrics = _build_metrics(num_classes, list(self.cfg.infer.metrics["items"]))
		for metric in stream_metrics.values():
			metric.to(self.device)
		self._metric_states[stream_name] = stream_metrics

	def _update_metrics(self, stream_name: str, probs: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
		for name, metric in self._metric_states[stream_name].items():
			if name == "ece":
				metric.update(probs, targets)
			else:
				metric.update(preds, targets)

	def _append_records(
		self,
		image_ids: Sequence[Any],
		fold: Sequence[Any],
		targets: torch.Tensor,
		preds: torch.Tensor,
		probs: torch.Tensor,
		uncertainty: Optional[torch.Tensor],
		stream_name: str,
	) -> None:
		confs = probs.max(dim=1).values

		targets_cpu = targets.detach().cpu().tolist()
		preds_cpu = preds.detach().cpu().tolist()
		confs_cpu = confs.detach().cpu().tolist()
		probs_cpu = probs.detach().cpu().tolist()

		unc_cpu = None
		if uncertainty is not None:
			unc_cpu = _to_cpu_tensor(uncertainty).view(-1).tolist()

		for idx in range(len(image_ids)):
			record = {
				"image_id": str(image_ids[idx]),
				"fold": str(fold[idx]) if fold is not None else "unknown",
				"target": int(targets_cpu[idx]),
				"prediction": int(preds_cpu[idx]),
				"confidence": float(confs_cpu[idx]),
				"class_probs": json.dumps(probs_cpu[idx]),
				"stream": stream_name,
			}
			if unc_cpu is not None:
				record["uncertainty"] = float(unc_cpu[idx])
			self._records.append(record)

	def _finalize(self) -> Dict[str, Any]:
		metrics: Dict[str, Any] = {}

		if bool(self.cfg.infer.metrics.enabled) and not self._skip_metrics:
			records_df = pd.DataFrame(self._records) if self._records else None
			compute_nll = "nll" in list(self.cfg.infer.metrics["items"])

			for stream_name, stream_metrics in self._metric_states.items():
				for name, metric in stream_metrics.items():
					value = metric.compute()
					metrics[f"{stream_name}.{name}"] = float(value.detach().cpu().item())

				if compute_nll and records_df is not None:
					stream_df = records_df[records_df["stream"] == stream_name]
					if not stream_df.empty:
						probs_t = torch.tensor(stream_df["class_probs"].map(json.loads).tolist(), dtype=torch.float32)
						targets_t = torch.tensor(stream_df["target"].tolist(), dtype=torch.long)
						nll = torch.nn.functional.nll_loss(torch.log(probs_t + 1e-8), targets_t)
						metrics[f"{stream_name}.nll"] = float(nll.item())
		elif self._skip_metrics:
			logger.warning(self._skip_metrics_reason)

		if bool(self.cfg.infer.save.save_csv):
			df = pd.DataFrame(self._records)
			csv_path = self.output_root / "predictions.csv"
			df.to_csv(csv_path, index=False)
			logger.info(f"Saved predictions to {csv_path}")

		if bool(self.cfg.infer.save.save_metrics_json):
			metrics_path = self.output_root / "metrics.json"
			with open(metrics_path, "w", encoding="utf-8") as f:
				json.dump(metrics, f, indent=2)
			logger.info(f"Saved metrics to {metrics_path}")

		return metrics

	@torch.no_grad()
	def run(self) -> Dict[str, Any]:
		saved_images = 0
		image_dir = self.output_root / "images"
		if bool(self.cfg.infer.save.save_images):
			image_dir.mkdir(parents=True, exist_ok=True)

		for batch in self.dataloader:
			image_ids = batch["image_id"]
			fold = batch.get("fold", ["test"] * len(image_ids))
			targets = batch["target"].to(self.device)

			stream_tensors = {
				"real": batch["real_image"].to(self.device),
				"artifact": batch["artifact_simulated_image"].to(self.device),
			}

			for stream_name, x in stream_tensors.items():
				logits, probs, uncertainty = _extract_logits_probs(
					model=self.model,
					x=x,
					use_mc_dropout=bool(self.cfg.infer.model.use_mc_dropout),
					mc_passes=int(self.cfg.infer.model.mc_passes),
				)
				preds = torch.argmax(probs, dim=1)
				num_classes = probs.shape[1]

				self._check_metric_compatibility(targets=targets, output_num_classes=num_classes)

				if bool(self.cfg.infer.metrics.enabled) and not self._skip_metrics:
					self._ensure_metrics(num_classes=num_classes, stream_name=stream_name)
					self._update_metrics(stream_name=stream_name, probs=probs, preds=preds, targets=targets)

				self._append_records(
					image_ids=image_ids,
					fold=fold,
					targets=targets,
					preds=preds,
					probs=probs,
					uncertainty=uncertainty,
					stream_name=stream_name,
				)

			if bool(self.cfg.infer.save.save_images) and saved_images < int(self.cfg.infer.save.max_images_to_save):
				batch_size = batch["real_image"].shape[0]
				budget = int(self.cfg.infer.save.max_images_to_save) - saved_images
				limit = min(batch_size, budget)
				for i in range(limit):
					image_id = str(image_ids[i]).replace("/", "_")
					real_path = image_dir / f"{image_id}_real.png"
					art_path = image_dir / f"{image_id}_artifact.png"
					_save_tensor_image(batch["real_image"][i], real_path)
					_save_tensor_image(batch["artifact_simulated_image"][i], art_path)
				saved_images += limit

		return self._finalize()
