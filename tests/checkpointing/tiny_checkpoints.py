"""Test-only builders for tiny, untrained-but-valid v2 checkpoints of each family.

Written the same way `scripts/ensemble/assemble_ensemble_checkpoint.py` writes one
(`state_dict` + `hyper_parameters` + `sngp_core`), so they load through every path in
`src/checkpointing/io.py` without a training run. No network: `pretrained=False`.
"""
from pathlib import Path
from typing import Dict

import torch

from src.checkpointing.spec import build_meta
from src.models.baseline.baseline_models import BaselineClassifier
from src.models.baseline_lit_module import BaselineLitModule
from src.models.deep_ensemble_lit_module import DeepEnsembleLitModule
from src.models.ensemble.deep_ensemble_model import DeepEnsemble
from src.models.sngp.sngp_classifier import SNGPClassifier
from src.models.sngp_lit_module import SNGPLitModule


class FakeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 8, num_classes: int = 3, seed: int = 0):
        self.n, self.num_classes = n, num_classes
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 224, 224, generator=gen)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return f"img{idx}", self.images[idx], idx % self.num_classes, "val"


def make_loader(n: int = 8, num_classes: int = 3, batch_size: int = 4):
    return torch.utils.data.DataLoader(FakeClassificationDataset(n=n, num_classes=num_classes), batch_size=batch_size)


def _idx_to_class(num_classes: int) -> Dict[int, str]:
    return {i: f"class_{i}" for i in range(num_classes)}


def _save(lit_module, net, path: Path, num_classes: int, extra_meta: Dict = None) -> Path:
    meta = build_meta(lit_module, net_spec=net.spec, num_classes=num_classes,
                      idx_to_class=_idx_to_class(num_classes), dataset_name="fake")
    if extra_meta:
        meta.update(extra_meta)
    checkpoint = {
        "state_dict": lit_module.state_dict(),
        "hyper_parameters": dict(lit_module.hparams),
        "sngp_core": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(path))
    return path


def save_baseline_checkpoint(path: Path, num_classes: int = 3, temperature: float = 1.0, extra_meta: Dict = None) -> Path:
    torch.manual_seed(0)
    net = BaselineClassifier(arch="resnet18", num_classes=num_classes, pretrained=False, dropout_p=0.2, temperature=temperature)
    lit = BaselineLitModule(net=net, num_classes=num_classes)
    return _save(lit, net, Path(path), num_classes, extra_meta)


def save_sngp_checkpoint(path: Path, num_classes: int = 3, rff_dim: int = 32, mean_field: bool = True) -> Path:
    torch.manual_seed(0)
    net = SNGPClassifier(num_classes=num_classes, arch="resnet18", pretrained=False, rff_dim=rff_dim, mean_field=mean_field)
    lit = SNGPLitModule(net=net, num_classes=num_classes)
    # Two train-mode forwards populate `precision_accum`, exactly as one training epoch would.
    net.train()
    with torch.no_grad():
        net(torch.randn(8, 3, 224, 224))
        net(torch.randn(8, 3, 224, 224))
    net.eval()
    return _save(lit, net, Path(path), num_classes)


def save_deep_ensemble_checkpoint(path: Path, num_classes: int = 3, num_estimators: int = 2) -> Path:
    torch.manual_seed(0)
    spec = {"name": "baseline_classifier", "arch": "resnet18", "num_classes": num_classes, "pretrained": False, "dropout_p": 0.2}
    net = DeepEnsemble(spec, num_estimators=num_estimators)
    lit = DeepEnsembleLitModule(net=net, num_classes=num_classes, num_estimators=num_estimators)
    return _save(lit, net, Path(path), num_classes)
