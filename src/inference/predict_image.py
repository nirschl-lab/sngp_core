"""Thin, real single/batch-image inference helper built on the same canonical
checkpoint loader as everything else (`src/checkpointing/io.py`).

This is deliberately small: for anything beyond "run this net on these images" --
dataset-wide inference with metrics, artifact-paired evaluation, CSV/JSON output --
use the single Hydra entrypoint `src/inference/infer.py` instead (see
docs/INFERENCE_GUIDE.md). This module exists only because a single-image code path is
a legitimate, common enough need to deserve a documented helper rather than yet
another parallel inference API.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from torchvision import transforms

from src.checkpointing.io import load_net

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_as_tensor(image_path: Union[str, Path], transform=None) -> torch.Tensor:
    """Load and preprocess a single image to a `[1, 3, 224, 224]` tensor."""
    transform = transform or _DEFAULT_TRANSFORM
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict_image(
    ckpt_path: Union[str, Path],
    image_path: Union[str, Path],
    *,
    class_names: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    use_mc_dropout: bool = False,
    mc_passes: int = 25,
) -> Dict:
    """Run one checkpoint on one image. Returns predicted class, confidence, and the
    full probability vector; adds `uncertainty` whenever the net reports it -- MC-Dropout
    predictive std when `use_mc_dropout=True` and the net supports it
    (`BaselineClassifier.mc_predict`), otherwise `ModelOutput.variance` (SNGP predictive
    variance / ensemble disagreement)."""
    net = load_net(ckpt_path, device=device)
    x = load_image_as_tensor(image_path).to(next(net.parameters()).device)

    with torch.no_grad():
        if use_mc_dropout and hasattr(net, "mc_predict"):
            _, probs, std = net.mc_predict(x, T=mc_passes, return_std=True, apply_softmax=True)
            uncertainty = std.mean(dim=1)
        else:
            output = net(x)
            probs = torch.softmax(output.logits, dim=1)
            # SNGP predictive variance / ensemble disagreement, when the net reports it.
            uncertainty = output.variance.squeeze(-1) if output.variance is not None else None

    pred_class = int(probs.argmax(dim=1).item())
    result = {
        "predicted_class": pred_class,
        "predicted_label": class_names[pred_class] if class_names else str(pred_class),
        "confidence": float(probs.max(dim=1).values.item()),
        "probs": probs.squeeze(0).cpu().numpy(),
    }
    if uncertainty is not None:
        result["uncertainty"] = float(uncertainty.item())
    return result


def predict_batch(
    ckpt_path: Union[str, Path],
    image_paths: Sequence[Union[str, Path]],
    *,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict:
    """Run one checkpoint over a list of image paths. Returns per-image predictions
    plus the stacked probability matrix `[N, num_classes]`, and `uncertainty` `[N]`
    whenever the net reports `ModelOutput.variance`."""
    net = load_net(ckpt_path, device=device)
    net_device = next(net.parameters()).device

    all_probs = []
    all_uncertainty = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch = torch.cat([load_image_as_tensor(p) for p in batch_paths], dim=0).to(net_device)
        with torch.no_grad():
            output = net(batch)
            probs = torch.softmax(output.logits, dim=1)
        all_probs.append(probs.cpu())
        if output.variance is not None:
            all_uncertainty.append(output.variance.squeeze(-1).cpu())

    probs = torch.cat(all_probs, dim=0)
    result = {
        "predictions": probs.argmax(dim=1).numpy(),
        "probs": probs.numpy(),
    }
    if all_uncertainty:
        result["uncertainty"] = torch.cat(all_uncertainty, dim=0).numpy()
    return result
