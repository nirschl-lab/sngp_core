"""Upload trained checkpoints to the `nirschl-lab/sngp-models` HF Hub repo via
`src/models/hf_loader.py::HFModelUploader` -- the raw-state_dict + config.json format
consumed by `HFModelLoader`/`quick_sngp_inference`/`quick_baseline_inference` (see the
project README's "quick inference" section). This is a lighter-weight, non-
`transformers`-dependent alternative to `scripts/hf/export_to_hub.py`'s
`trust_remote_code` bundles -- use that script instead if you want
`AutoModel.from_pretrained(..., trust_remote_code=True)` support.

Reads architecture identity straight from each checkpoint's own metadata (see
`src/checkpointing/io.py`) -- update MODEL_SPECS below to point at your checkpoints,
nothing else needs to be retyped by hand.

Before running:
1. huggingface-cli login
2. Update repo_id and MODEL_SPECS below

Usage:
    uv run scripts/upload_to_hf.py
"""
from pathlib import Path
from typing import List, TypedDict

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.checkpointing.io import load_net, read_meta  # noqa: E402
from src.models.hf_loader import HFModelUploader  # noqa: E402


class ModelSpec(TypedDict):
    name: str
    ckpt_path: str


MODEL_SPECS: List[ModelSpec] = [
    {"name": "acevedo_baseline_resnet18", "ckpt_path": "logs/train/runs/<run>/checkpoints/best.ckpt"},
    {"name": "wong_baseline_resnet18", "ckpt_path": "logs/train/runs/<run>/checkpoints/best.ckpt"},
    {"name": "acevedo_sngp_resnet18", "ckpt_path": "logs/train/runs/<run>/checkpoints/best.ckpt"},
    {"name": "wong_sngp_resnet18", "ckpt_path": "logs/train/runs/<run>/checkpoints/best.ckpt"},
]

_NET_SPEC_NAME_TO_MODEL_TYPE = {"baseline_classifier": "baseline", "sngp_classifier": "sngp"}


def upload_models_to_hf(repo_id: str = "nirschl-lab/sngp-models") -> None:
    uploader = HFModelUploader(repo_id=repo_id)

    for spec in MODEL_SPECS:
        ckpt_path = Path(spec["ckpt_path"])
        print(f"\n{'=' * 60}\nUploading: {spec['name']}\n{'=' * 60}")

        if not ckpt_path.exists():
            print(f"Checkpoint not found, skipping: {ckpt_path}")
            continue

        try:
            meta = read_meta(ckpt_path)
        except ValueError as exc:
            print(f"Skipping {ckpt_path}: {exc}")
            continue

        model_type = _NET_SPEC_NAME_TO_MODEL_TYPE.get(meta.net_spec.get("name"))
        if model_type is None:
            print(f"Skipping {ckpt_path}: HF upload only supports baseline/SNGP nets, got {meta.net_spec.get('name')!r}")
            continue

        model = load_net(ckpt_path, device="cpu")
        config_dict = {k: v for k, v in meta.net_spec.items() if k != "name"}

        commit_info = uploader.upload_model(
            model=model,
            model_name=spec["name"],
            model_type=model_type,
            config_dict=config_dict,
            commit_message=f"Upload {spec['name']} ({model_type})",
        )
        print(f"Uploaded: {commit_info.commit_url}")

    print(f"\nDone. Repository: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    upload_models_to_hf()
