"""Checkpoint-path resolution: turns a `wandb-artifact://<artifact>` URI into a local
`.ckpt` path (downloading it if needed), or passes a plain path through unchanged.

Extracted from `src/eval.py` so `src/inference/infer.py` gets the same capability
instead of only supporting local paths.
"""
from typing import Any, List, Optional


def resolve_ckpt_uri(
    ckpt_path: str,
    *,
    loggers: Optional[List[Any]] = None,
    wandb_project: Optional[str] = None,
    wandb_job_type: str = "checkpoint-resolve",
) -> str:
    if not ckpt_path.startswith("wandb-artifact://"):
        return ckpt_path

    import wandb
    from lightning.pytorch.loggers import WandbLogger

    wandb_logger = next((lg for lg in (loggers or []) if isinstance(lg, WandbLogger)), None)
    if wandb_logger is not None:
        run = wandb_logger.experiment
    else:
        run = wandb.init(project=wandb_project or "default-project", job_type=wandb_job_type)

    artifact = run.use_artifact(ckpt_path.replace("wandb-artifact://", ""), type="model")
    artifact_dir = artifact.download()
    return f"{artifact_dir}/model.ckpt"
