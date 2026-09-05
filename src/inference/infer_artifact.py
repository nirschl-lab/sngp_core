from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.inference.infer import _resolve_output_root
from src.inference.records import (
	build_metrics,
	build_records,
	extract_model_outputs,
	resolve_device,
	write_outputs,
)
from src.metrics.brier import brier_score


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


class ArtifactInferenceRunner:
	"""Artifact-specific inference runner for paired real/artifact images."""

	def __init__(
		self,
		model: torch.nn.Module,
		dataloader: DataLoader,
		cfg: DictConfig,
		class_names: Optional[Dict[int, str]] = None,
		expected_num_classes: Optional[int] = None,
		default_run_name: str = "",
		provenance: Optional[Dict[str, Any]] = None,
	) -> None:
		self.model = model
		self.dataloader = dataloader
		self.cfg = cfg
		self.class_names = class_names or {}
		self.expected_num_classes = expected_num_classes
		self.provenance = provenance

		self.device = resolve_device(str(self.cfg.infer.runtime.device))
		self.model.to(self.device)
		self.model.eval()

		self.output_root = _resolve_output_root(cfg, default_run_name)

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
		stream_metrics = build_metrics(num_classes, list(self.cfg.infer.metrics["items"]))
		for metric in stream_metrics.values():
			metric.to(self.device)
		self._metric_states[stream_name] = stream_metrics

	def _update_metrics(self, stream_name: str, probs: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
		for name, metric in self._metric_states[stream_name].items():
			if name in ("ece", "auroc", "auprc"):
				metric.update(probs, targets)
			else:
				metric.update(preds, targets)

	def _finalize(self) -> Dict[str, Any]:
		metrics: Dict[str, Any] = {}

		if bool(self.cfg.infer.metrics.enabled) and not self._skip_metrics:
			records_df = pd.DataFrame(self._records) if self._records else None
			items = list(self.cfg.infer.metrics["items"])
			compute_nll = "nll" in items
			compute_brier = "brier" in items

			for stream_name, stream_metrics in self._metric_states.items():
				for name, metric in stream_metrics.items():
					value = metric.compute()
					metrics[f"{stream_name}.{name}"] = float(value.detach().cpu().item())

				if (compute_nll or compute_brier) and records_df is not None:
					stream_df = records_df[records_df["stream"] == stream_name]
					if not stream_df.empty:
						probs_t = torch.tensor(stream_df["class_probs"].map(json.loads).tolist(), dtype=torch.float32)
						targets_t = torch.tensor(stream_df["target"].tolist(), dtype=torch.long)
						if compute_nll:
							nll = torch.nn.functional.nll_loss(torch.log(probs_t + 1e-8), targets_t)
							metrics[f"{stream_name}.nll"] = float(nll.item())
						if compute_brier:
							metrics[f"{stream_name}.brier"] = brier_score(
								probs_t, targets_t, num_classes=probs_t.shape[1]
							)
		elif self._skip_metrics:
			logger.warning(self._skip_metrics_reason)

		write_outputs(
			output_root=self.output_root,
			records=self._records,
			metrics=metrics,
			save_cfg=self.cfg.infer.save,
			provenance=self.provenance,
		)

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
				outputs = extract_model_outputs(
					model=self.model,
					x=x,
					use_mc_dropout=bool(self.cfg.infer.runtime.use_mc_dropout),
					mc_passes=int(self.cfg.infer.runtime.mc_passes),
				)
				preds = torch.argmax(outputs.probs, dim=1)
				num_classes = outputs.probs.shape[1]

				self._check_metric_compatibility(targets=targets, output_num_classes=num_classes)

				if bool(self.cfg.infer.metrics.enabled) and not self._skip_metrics:
					self._ensure_metrics(num_classes=num_classes, stream_name=stream_name)
					self._update_metrics(
						stream_name=stream_name, probs=outputs.probs, preds=preds, targets=targets
					)

				self._records.extend(
					build_records(
						image_ids=image_ids,
						fold=fold,
						targets=targets,
						preds=preds,
						outputs=outputs,
						stream_name=stream_name,
					)
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
