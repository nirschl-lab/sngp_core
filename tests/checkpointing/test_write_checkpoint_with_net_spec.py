"""`src/checkpointing/io.py::write_checkpoint_with_net_spec` -- the sanctioned way to bake a
post-hoc-fitted knob into a checkpoint -- and `read_meta`'s forward compatibility."""
import pytest
import torch

from src.checkpointing.io import load_lit_module, load_net, read_meta, write_checkpoint_with_net_spec
from tests.checkpointing.tiny_checkpoints import save_baseline_checkpoint

RECORD = {"knob": "temperature", "value": 1.73, "previous_value": 1.0, "split": "val", "objective": "nll"}


def test_writes_sibling_with_both_spec_copies_updated_and_weights_intact(tmp_path):
    src = save_baseline_checkpoint(tmp_path / "best.ckpt")
    dst = write_checkpoint_with_net_spec(src, tmp_path / "best.calibrated.ckpt", {"temperature": 1.73}, calibration=RECORD)
    assert dst == tmp_path / "best.calibrated.ckpt"

    meta = read_meta(dst)
    assert meta.net_spec["temperature"] == 1.73
    assert meta.calibration == RECORD
    assert meta.idx_to_class == read_meta(src).idx_to_class

    # `load_net` rebuilds from sngp_core.net_spec ...
    net_dst = load_net(dst, device="cpu")
    assert net_dst.temperature == 1.73
    # ... and `load_lit_module` from hyper_parameters.net_spec -- both must see the knob.
    lit_dst = load_lit_module(dst, device="cpu")
    assert lit_dst.net.temperature == 1.73
    assert lit_dst.hparams["net_spec"]["temperature"] == 1.73

    # Weights are copied verbatim.
    net_src = load_net(src, device="cpu")
    for key, value in net_src.state_dict().items():
        assert torch.equal(value, net_dst.state_dict()[key]), key

    # The source is never modified.
    src_meta = read_meta(src)
    assert src_meta.net_spec["temperature"] == 1.0
    assert src_meta.calibration is None


def test_rejects_unknown_key_name_and_non_json_values(tmp_path):
    src = save_baseline_checkpoint(tmp_path / "best.ckpt")
    with pytest.raises(ValueError, match="not a constructor argument"):
        write_checkpoint_with_net_spec(src, tmp_path / "out.ckpt", {"kernel_amplitude": 2.0})
    with pytest.raises(ValueError, match="`name`"):
        write_checkpoint_with_net_spec(src, tmp_path / "out.ckpt", {"name": "sngp_classifier"})
    with pytest.raises(ValueError, match="JSON"):
        write_checkpoint_with_net_spec(src, tmp_path / "out.ckpt", {"temperature": torch.tensor(2.0)})
    with pytest.raises(ValueError, match="no net_spec updates"):
        write_checkpoint_with_net_spec(src, tmp_path / "out.ckpt", {})
    assert not (tmp_path / "out.ckpt").exists()


def test_refuses_in_place_and_existing_destination_unless_overwrite(tmp_path):
    src = save_baseline_checkpoint(tmp_path / "best.ckpt")
    with pytest.raises(ValueError, match="in place"):
        write_checkpoint_with_net_spec(src, src, {"temperature": 2.0})

    dst = write_checkpoint_with_net_spec(src, tmp_path / "out.ckpt", {"temperature": 2.0})
    with pytest.raises(FileExistsError):
        write_checkpoint_with_net_spec(src, dst, {"temperature": 3.0})
    write_checkpoint_with_net_spec(src, dst, {"temperature": 3.0}, overwrite=True)
    assert read_meta(dst).net_spec["temperature"] == 3.0


def test_read_meta_tolerates_unknown_metadata_keys_and_missing_calibration(tmp_path):
    src = save_baseline_checkpoint(tmp_path / "best.ckpt", extra_meta={"future_key": {"x": 1}})
    meta = read_meta(src)
    assert meta.calibration is None
    assert not hasattr(meta, "future_key")
