from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import hydra
from hydra import compose, initialize_config_dir
import numpy as np
import pandas as pd
import rootutils
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    Accuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.outputs import ModelOutput  # noqa: E402


DEFAULT_INFER_RUNTIME_CFG: Dict[str, Any] = {
    "device": "cuda",
    "batch_size_override": None,
    "strict": False,
    "use_mc_dropout": False,
    "mc_passes": 10,
}

DEFAULT_INFER_METRICS_CFG: Dict[str, Any] = {
    "enabled": True,
    "items": [
        "acc",
        "ece",
        "precision",
        "recall",
        "f1",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "nll",
        "auroc",
        "auprc",
    ],
}

DEFAULT_INFER_SAVE_CFG: Dict[str, Any] = {
    "run_name": "",
    "save_csv": True,
    "save_metrics_json": True,
    "save_images": False,
    "max_images_to_save": 64,
}

_RUN_ID_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")


def _extract_ckpt_run_id(ckpt_path: Union[str, Path]) -> str:
    """This project's run-id stamp (`${now:%Y-%m-%d}_${now:%H-%M-%S}`, see
    `configs/hydra/default.yaml`) if present in `ckpt_path`, else the current time in
    the same format."""
    match = _RUN_ID_PATTERN.search(str(ckpt_path))
    if match:
        return match.group(0)
    fallback = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.warning(f"No run-id timestamp found in ckpt_path={ckpt_path!r}; using current time {fallback!r}.")
    return fallback


def derive_default_run_name(
    ckpt_path: Union[str, Path],
    net_name: str,
    data_name: str,
    use_mc_dropout: bool = False,
) -> str:
    """Default `infer.save.run_name` when left blank: '<net_spec name>/<ckpt_run_id>/<dataset>'.

    `net_name` is `read_meta(ckpt_path).net_spec["name"]` -- the `NET_REGISTRY` key,
    which is NOT the same as `configs/model/*.yaml`'s `name:` field (they differ for
    deep ensembles: "deep_ensemble" vs "deep_ensemble_classifier"). When
    `use_mc_dropout` is set, `_mcdropout` is appended to the net name so MC-Dropout
    runs don't overwrite a plain inference run's `predictions.csv`/`metrics.json` in
    the same `<net_name>/<ckpt_run_id>/<dataset>` folder.
    """
    if use_mc_dropout:
        net_name = f"{net_name}_mcdropout"
    return f"{net_name}/{_extract_ckpt_run_id(ckpt_path)}/{data_name}"


def _resolve_output_root(cfg: DictConfig, default_run_name: str) -> Path:
    """Shared `save_path`/`run_name` resolution used by both inference runners."""
    save_root = Path(str(cfg.save_path))
    run_name = str(cfg.infer.save.run_name) or default_run_name
    output_root = save_root / run_name if run_name else save_root
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _resolve_sections(cfg: DictConfig) -> DictConfig:
    if "infer" not in cfg:
        cfg.infer = OmegaConf.create({})

    cfg.infer.runtime = OmegaConf.merge(
        OmegaConf.create(DEFAULT_INFER_RUNTIME_CFG),
        cfg.infer.get("runtime") or OmegaConf.create({}),
    )
    cfg.infer.metrics = OmegaConf.merge(
        OmegaConf.create(DEFAULT_INFER_METRICS_CFG),
        cfg.infer.get("metrics") or OmegaConf.create({}),
    )
    cfg.infer.save = OmegaConf.merge(
        OmegaConf.create(DEFAULT_INFER_SAVE_CFG),
        cfg.infer.get("save") or OmegaConf.create({}),
    )
    return cfg


def _normalize_fold(fold: str) -> str:
    fold_norm = fold.strip().lower()
    if fold_norm == "val":
        return "validation"
    if fold_norm in {"train", "validation", "test", "all"}:
        return fold_norm
    raise ValueError(f"Unsupported fold '{fold}'. Use train, val, test, or all.")


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


def _extract_logits_probs(
    model: torch.nn.Module,
    x: torch.Tensor,
    use_mc_dropout: bool,
    mc_passes: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return (logits, probs, uncertainty)."""

    if use_mc_dropout and hasattr(model, "mc_predict"):
        result = model.mc_predict(x, T=mc_passes, return_std=True, apply_softmax=True)
        if isinstance(result, tuple) and len(result) == 3:
            logits, probs, uncertainty = result
            return logits, probs, uncertainty

    output = model(x)

    uncertainty = None
    if isinstance(output, ModelOutput):
        logits = output.logits
        uncertainty = output.variance
    elif isinstance(output, tuple):
        # Legacy nets returning a bare (mean_field_logits, raw_logits, pred_var) tuple.
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
    if "precision_micro" in metric_names:
        metrics["precision_micro"] = MulticlassPrecision(num_classes=num_classes, average="micro")
    if "recall_micro" in metric_names:
        metrics["recall_micro"] = MulticlassRecall(num_classes=num_classes, average="micro")
    if "f1_micro" in metric_names:
        metrics["f1_micro"] = MulticlassF1Score(num_classes=num_classes, average="micro")
    if "auroc" in metric_names:
        metrics["auroc"] = MulticlassAUROC(num_classes=num_classes, average="macro")
    if "auprc" in metric_names:
        metrics["auprc"] = MulticlassAveragePrecision(num_classes=num_classes, average="macro")

    return metrics


class BaseInferenceRunner:
    """Framework-agnostic inference wrapper around a torch model and dataloader."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        cfg: DictConfig,
        class_names: Optional[Dict[int, str]] = None,
        expected_num_classes: Optional[int] = None,
        default_run_name: str = "",
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg
        self.class_names = class_names or {}
        self.expected_num_classes = expected_num_classes

        self.device = _resolve_device(str(self.cfg.infer.runtime.device))
        self.model.to(self.device)
        self.model.eval()

        self.output_root = _resolve_output_root(cfg, default_run_name)

        self._records: List[Dict[str, Any]] = []
        self._metric_states: Dict[str, Any] = {}
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

    def _ensure_metrics(self, num_classes: int) -> None:
        if self._metric_states:
            return
        self._metric_states = _build_metrics(num_classes, list(self.cfg.infer.metrics["items"]))
        for metric in self._metric_states.values():
            metric.to(self.device)

    def _update_metrics(self, probs: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> None:
        for name, metric in self._metric_states.items():
            if name in ("ece", "auroc", "auprc"):
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
            for name, metric in self._metric_states.items():
                value = metric.compute()
                metrics[name] = float(value.detach().cpu().item())

            if "nll" in list(self.cfg.infer.metrics["items"]) and self._records:
                records_df = pd.DataFrame(self._records)
                probs = records_df["class_probs"].map(json.loads).to_list()
                probs_t = torch.tensor(probs, dtype=torch.float32)
                targets_t = torch.tensor(records_df["target"].tolist(), dtype=torch.long)
                nll = torch.nn.functional.nll_loss(torch.log(probs_t + 1e-8), targets_t)
                metrics["nll"] = float(nll.item())
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
        raise NotImplementedError("Subclasses must implement run().")


class ClassificationInferenceRunner(BaseInferenceRunner):
    @torch.no_grad()
    def run(self) -> Dict[str, Any]:
        for batch in self.dataloader:
            image_ids, x, targets, fold = batch
            x = x.to(self.device)
            targets = targets.to(self.device)

            logits, probs, uncertainty = _extract_logits_probs(
                model=self.model,
                x=x,
                use_mc_dropout=bool(self.cfg.infer.runtime.use_mc_dropout),
                mc_passes=int(self.cfg.infer.runtime.mc_passes),
            )
            if not torch.isfinite(logits).all():
                raise RuntimeError("Encountered non-finite logits during inference.")
            if not torch.isfinite(probs).all():
                raise RuntimeError("Encountered non-finite probabilities during inference.")
            preds = torch.argmax(probs, dim=1)

            num_classes = probs.shape[1]
            self._check_metric_compatibility(targets=targets, output_num_classes=num_classes)

            if bool(self.cfg.infer.metrics.enabled) and not self._skip_metrics:
                self._ensure_metrics(num_classes=num_classes)
                self._update_metrics(probs=probs, preds=preds, targets=targets)

            self._append_records(
                image_ids=image_ids,
                fold=fold,
                targets=targets,
                preds=preds,
                probs=probs,
                uncertainty=uncertainty,
                stream_name="default",
            )

        return self._finalize()



def _instantiate_datamodule(cfg: DictConfig):
    test_augmentations = None
    if cfg.data.get("img_augmentations") and cfg.data.img_augmentations.get("test"):
        test_augmentations = hydra.utils.instantiate(cfg.data.img_augmentations.test)

    dm_cfg = cfg.data.get("datamodule")
    if not isinstance(dm_cfg, DictConfig):
        dm_cfg = OmegaConf.create(dm_cfg)

    if cfg.infer.runtime.batch_size_override is not None:
        dm_cfg.batch_size = cfg.infer.runtime.batch_size_override

    datamodule = hydra.utils.instantiate(
        dm_cfg,
        train_augmentations=None,
        val_augmentations=None,
        test_augmentations=test_augmentations,
    )
    return datamodule


def _instantiate_model(cfg: DictConfig, device: torch.device):
    """Load the net directly from the checkpoint via the single canonical loader.

    The checkpoint is authoritative for architecture identity (see
    `src/checkpointing/io.py`) -- no separate `configs/infer/model/*.yaml` lookup or
    hparams-merging is needed; a mismatched/unavailable Hydra model config can no
    longer produce a checkpoint/model compatibility error, because there is no longer
    a second, independently-configured model in the loop to be incompatible with.

    Returns `(model, meta)` -- the caller reuses `meta` (already-loaded, no extra
    unpickling) for both the `idx_to_class` fallback and the default inference
    output-folder name.
    """
    from src.checkpointing.io import load_net, read_meta

    meta = read_meta(cfg.ckpt_path)
    logger.info(
        f"Loading checkpoint: lit_module={meta.lit_module}, net={meta.net_spec.get('name')}, "
        f"arch={meta.net_spec.get('arch')}, num_classes={meta.num_classes}"
    )
    model = load_net(
        cfg.ckpt_path,
        device=str(device),
        strict=bool(cfg.infer.runtime.strict),
    )
    return model, meta


def run_inference(cfg: DictConfig) -> Dict[str, Any]:
    if not cfg.get("ckpt_path"):
        raise ValueError("ckpt_path must be set for inference")

    cfg = _resolve_sections(cfg)
    fold = _normalize_fold(str(cfg.get("fold", "test")))

    device = _resolve_device(str(cfg.infer.runtime.device))

    datamodule = _instantiate_datamodule(cfg)

    if fold == "all":
        if hasattr(datamodule, "test_all_folds"):
            datamodule.test_all_folds = True
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()
    elif fold == "train":
        datamodule.setup(stage="fit")
        dataloader = datamodule.train_dataloader()
    elif fold == "validation":
        datamodule.setup(stage="fit")
        dataloader = datamodule.val_dataloader()
    else:
        datamodule.setup(stage="test")
        dataloader = datamodule.test_dataloader()

    model, ckpt_meta = _instantiate_model(cfg, device=device)

    class_names = {}
    if hasattr(datamodule, "trainer") and datamodule.trainer is not None:
        class_names = getattr(datamodule.trainer, "test_idx_to_classes", {})
    if not class_names and ckpt_meta.idx_to_class:
        class_names = ckpt_meta.idx_to_class

    default_run_name = derive_default_run_name(
        ckpt_path=cfg.ckpt_path,
        net_name=str(ckpt_meta.net_spec.get("name", "model")),
        data_name=str(cfg.data.name),
        use_mc_dropout=bool(cfg.infer.runtime.use_mc_dropout),
    )

    expected_num_classes = None
    if hasattr(datamodule, "num_classes"):
        dm_num_classes = getattr(datamodule, "num_classes")
        if isinstance(dm_num_classes, int):
            expected_num_classes = dm_num_classes

    if dataloader is None:
        raise RuntimeError("Datamodule returned no test dataloader.")

    # import pdb; pdb.set_trace()
    try:
        first_batch = next(iter(dataloader))
    except StopIteration as exc:
        raise RuntimeError("Dataloader is empty for the selected fold.") from exc

    if fold == "all":
        dataloader = datamodule.test_dataloader()
    elif fold == "train":
        dataloader = datamodule.train_dataloader()
    elif fold == "validation":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    if isinstance(first_batch, Mapping) and "artifact_simulated_image" in first_batch:
        logger.info("Detected artifact dataloader format. Using ArtifactInferenceRunner.")
        from src.inference.infer_artifact import ArtifactInferenceRunner

        runner = ArtifactInferenceRunner(
            model=model,
            dataloader=dataloader,
            cfg=cfg,
            class_names=class_names,
            expected_num_classes=expected_num_classes,
            default_run_name=default_run_name,
        )
    else:
        logger.info("Detected standard classification dataloader format. Using ClassificationInferenceRunner.")
        runner = ClassificationInferenceRunner(
            model=model,
            dataloader=dataloader,
            cfg=cfg,
            class_names=class_names,
            expected_num_classes=expected_num_classes,
            default_run_name=default_run_name,
        )

    metrics = runner.run()
    logger.info(f"Inference completed. Metrics: {metrics}")
    return metrics


def _compose_cfg(config_name: str = "infer", overrides: Optional[Iterable[str]] = None) -> DictConfig:
    config_dir = str((Path(__file__).resolve().parents[2] / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=list(overrides or []), return_hydra_config=False)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone inference entrypoint.")
    parser.add_argument(
        "--config-name",
        default="infer",
        help="Hydra-style config name inside configs/, without .yaml suffix.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Directory where predictions.csv, metrics.json, and optional images will be written.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any OmegaConf dotlist overrides, e.g. ckpt_path=... data=acevedo infer.model=sngp_classifier",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.overrides or [])
    if args.save_path is not None:
        overrides.append(f"save_path={args.save_path}")

    cfg = _compose_cfg(config_name=args.config_name, overrides=overrides)

    logger.info("Running standalone inference with separate infer config layer.")
    run_inference(cfg)


if __name__ == "__main__":
    main()
