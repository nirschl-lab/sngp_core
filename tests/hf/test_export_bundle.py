"""Verifies scripts/hf/export_to_hub.py produces a genuinely self-contained
`trust_remote_code` bundle: no `src.*` imports, and loading it via
`transformers.AutoModel.from_pretrained` reproduces the original checkpoint's outputs
exactly (this is what caught a real transformers/spectral_norm state-dict-key-fixup
interaction during development -- see the `_fix_state_dict_key_on_load` override in
the export script).
"""
import ast

import pytest
import torch

pytest.importorskip("transformers")

from scripts.hf.export_to_hub import export
from src.checkpointing.io import load_net
from tests.checkpointing.test_roundtrip import _build_and_train_one_step


def _assert_no_src_imports(module_path):
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            pytest.fail(f"{module_path} imports from {node.module!r} -- not self-contained")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    pytest.fail(f"{module_path} imports {alias.name!r} -- not self-contained")


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name,overrides,modeling_file",
    [
        ("baseline_classifier", ["model.net.pretrained=false"], "modeling_baseline_classifier.py"),
        ("sngp_classifier", ["model.net.pretrained=false"], "modeling_sngp_classifier.py"),
    ],
)
def test_export_bundle_is_self_contained_and_matches_original(tmp_path, model_name, overrides, modeling_file):
    _, ckpt_path, _ = _build_and_train_one_step(tmp_path / "train", model_name, overrides)

    out_dir = tmp_path / "export"
    export(str(ckpt_path), out_dir)

    modeling_path = out_dir / modeling_file
    assert modeling_path.exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "model.safetensors").exists()

    _assert_no_src_imports(modeling_path)

    from transformers import AutoModel

    hf_model = AutoModel.from_pretrained(str(out_dir), trust_remote_code=True)
    hf_model.eval()

    original_net = load_net(ckpt_path, device="cpu")
    original_net.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        hf_logits = hf_model(x)["logits"]
        original_logits = original_net(x).logits

    assert torch.allclose(hf_logits, original_logits, atol=1e-5)
