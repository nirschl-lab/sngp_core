"""End-to-end tests for scripts/checkpoints/calibrate_checkpoint.py on tiny checkpoints of
every family, with an in-memory fake loader (no network, no training)."""
from pathlib import Path

import pytest

from scripts.checkpoints.calibrate_checkpoint import calibrate, default_calibrated_path, knob_for, main
from src.checkpointing.io import load_net, read_meta
from tests.checkpointing.tiny_checkpoints import (
    make_loader,
    save_baseline_checkpoint,
    save_deep_ensemble_checkpoint,
    save_sngp_checkpoint,
)


def test_refuses_to_fit_on_the_test_split_before_any_io(tmp_path):
    missing = tmp_path / "does_not_exist.ckpt"
    assert main(["--ckpt", str(missing), "--experiment", "baseline_tang", "--split", "test"]) == 2


def test_default_calibrated_path_is_a_sibling():
    assert default_calibrated_path(Path("/x/checkpoints/best.ckpt")) == Path("/x/checkpoints/best.calibrated.ckpt")


def test_knob_for_each_family():
    assert knob_for({"name": "baseline_classifier"}) == "temperature"
    assert knob_for({"name": "deep_ensemble"}) == "temperature"
    assert knob_for({"name": "sngp_classifier"}) == "mean_field_factor"
    with pytest.raises(ValueError, match="No post-hoc calibration knob"):
        knob_for({"name": "something_else"})


def test_baseline_end_to_end_writes_temperature(tmp_path):
    src = save_baseline_checkpoint(tmp_path / "best.ckpt")
    value, out = calibrate(src, make_loader(), device="cpu", split="val", experiment="fake_exp", dataset_name="fake")

    assert out == tmp_path / "best.calibrated.ckpt"
    meta = read_meta(out)
    assert meta.net_spec["temperature"] == value > 0
    assert meta.calibration["knob"] == "temperature"
    assert meta.calibration["split"] == "val"
    assert meta.calibration["experiment"] == "fake_exp"
    assert meta.calibration["n_samples"] == 8
    assert meta.calibration["metrics_after"]["nll"] <= meta.calibration["metrics_before"]["nll"] + 1e-6
    assert meta.calibration["metrics_after"]["acc"] == meta.calibration["metrics_before"]["acc"]
    assert load_net(out, device="cpu").temperature == value
    # source untouched
    assert read_meta(src).net_spec["temperature"] == 1.0
    assert read_meta(src).calibration is None


def test_dry_run_and_report_only_write_nothing(tmp_path):
    src = save_baseline_checkpoint(tmp_path / "best.ckpt")
    value, out = calibrate(src, make_loader(), device="cpu", dry_run=True)
    assert out is None and value > 0
    previous, out = calibrate(src, make_loader(), device="cpu", report_only=True, split="test")
    assert out is None and previous == 1.0
    assert not (tmp_path / "best.calibrated.ckpt").exists()


def test_sngp_end_to_end_writes_mean_field_factor(tmp_path):
    src = save_sngp_checkpoint(tmp_path / "best.ckpt")
    value, out = calibrate(src, make_loader(), device="cpu", split="val")

    meta = read_meta(out)
    assert meta.calibration["knob"] == "mean_field_factor"
    assert meta.net_spec["mean_field_factor"] == value >= 0
    net = load_net(out, device="cpu")
    assert net.mean_field_factor == value
    assert net.gp_head.mean_field_factor == value
    assert meta.calibration["metrics_after"]["nll"] <= meta.calibration["metrics_before"]["nll"] + 1e-6


def test_sngp_ridge_sweep_writes_ridge_penalty_too(tmp_path):
    src = save_sngp_checkpoint(tmp_path / "best.ckpt")
    _, out = calibrate(src, make_loader(), device="cpu", ridges=[1e-3, 1.0])
    meta = read_meta(out)
    assert meta.net_spec["ridge_penalty"] in (1e-3, 1.0)
    assert meta.calibration["ridge_penalty"] == meta.net_spec["ridge_penalty"]
    assert load_net(out, device="cpu").gp_head.ridge == meta.net_spec["ridge_penalty"]


def test_sngp_with_mean_field_disabled_is_refused(tmp_path):
    src = save_sngp_checkpoint(tmp_path / "best.ckpt", mean_field=False)
    with pytest.raises(ValueError, match="mean_field=False"):
        calibrate(src, make_loader(), device="cpu")


def test_ridge_is_sngp_only(tmp_path):
    src = save_baseline_checkpoint(tmp_path / "best.ckpt")
    with pytest.raises(ValueError, match="only applies to SNGP"):
        calibrate(src, make_loader(), device="cpu", ridges=[1.0])


def test_deep_ensemble_writes_ensemble_level_temperature(tmp_path):
    src = save_deep_ensemble_checkpoint(tmp_path / "best.ckpt")
    value, out = calibrate(src, make_loader(), device="cpu")

    meta = read_meta(out)
    assert meta.calibration["knob"] == "temperature"
    assert meta.net_spec["temperature"] == value
    assert "temperature" not in meta.net_spec["base_model_spec"]
    net = load_net(out, device="cpu")
    assert net.temperature == value
    assert all(member.temperature == 1.0 for member in net.ensemble_members)
