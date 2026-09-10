from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import pdb
import cv2


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.checkpointing.io import read_meta
from src.checkpointing.resolve import resolve_ckpt_uri
from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _assert_model_cfg_matches_checkpoint(cfg: DictConfig, model: LightningModule) -> None:
    """Fail loudly when `cfg.model` describes a different architecture than the checkpoint.

    `eval.py` is config-authoritative (it Hydra-instantiates the model, then hands
    `ckpt_path` to `trainer.test`), while `src/inference/infer.py` is
    checkpoint-authoritative. That divergence is a real footgun: `configs/eval.yaml`
    defaults to `model: sngp_classifier` with `rff_dim: 1024`, but e.g.
    `configs/experiment/sngp_wong.yaml` trains with `rff_dim: 512`. Rather than
    restructure `eval.py`, compare the two specs up front and say exactly which keys
    disagree.
    """
    try:
        meta = read_meta(cfg.ckpt_path)
    except Exception as exc:  # unreadable/legacy checkpoint -- let trainer.test surface it
        log.warning(f"Could not read checkpoint metadata for a config/checkpoint consistency check: {exc}")
        return

    ckpt_spec = dict(meta.net_spec or {})
    cfg_spec = dict(getattr(model.net, "spec", {}) or {})
    if not ckpt_spec or not cfg_spec:
        return

    # `pretrained` only affects how weights were *initialized*; the state_dict overwrites
    # them either way, so a disagreement there is not a real mismatch.
    ignored = {"pretrained"}
    mismatched = {
        key: (cfg_spec.get(key), ckpt_spec.get(key))
        for key in set(ckpt_spec) | set(cfg_spec)
        if key not in ignored and cfg_spec.get(key) != ckpt_spec.get(key)
    }
    if mismatched:
        detail = "\n".join(f"  {key}: config={cfg!r}  checkpoint={ckpt!r}" for key, (cfg, ckpt) in sorted(mismatched.items()))
        raise ValueError(
            f"`cfg.model` does not match the architecture stored in {cfg.ckpt_path}:\n{detail}\n"
            "Re-run with the same `experiment=` override the checkpoint was trained with, "
            "or use `src/inference/infer.py`, which reads the architecture from the checkpoint."
        )

# OpenCV performance tweaks for albumentations
cv2.setNumThreads(0)
cv2.setUseOptimized(True)

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    if not cfg.ckpt_path.endswith(".ckpt"):
        wandb_cfg = cfg.get("logger", {}).get("wandb", {}) if cfg.get("logger") else {}
        cfg.ckpt_path = resolve_ckpt_uri(
            cfg.ckpt_path,
            loggers=logger,
            wandb_project=wandb_cfg.get("project", "default-project"),
            wandb_job_type=wandb_cfg.get("job_type", "inference"),
        )
        log.info(f"Using checkpoint from W&B artifact: {cfg.ckpt_path}")

    train_augmentations = None
    test_augmentations = None

    if cfg.data['img_augmentations']:
        log.info(f"Augmentations <{cfg.data.img_augmentations.test._target_}>")
        test_augmentations = hydra.utils.instantiate( cfg.data.img_augmentations.test)

    log.info(f"Instantiating datamodule <{cfg.data.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, train_augmentations=train_augmentations, test_augmentations=test_augmentations)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    _assert_model_cfg_matches_checkpoint(cfg, model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
