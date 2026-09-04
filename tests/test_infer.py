import json
import re
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.inference import infer, records
from src.models.outputs import ModelOutput


class DummyDataModule:
    def __init__(self, artifact_bank_dir: str, artifact_taxonomy_csv: str, **kwargs) -> None:
        self.artifact_bank_dir = artifact_bank_dir
        self.artifact_taxonomy_csv = artifact_taxonomy_csv


def test_infer_datamodule_resolves_paths_interpolations(monkeypatch):
    """The artifact datamodule's paths arrive as `${paths.*}` interpolations, which
    `_instantiate_datamodule` has to resolve -- an unresolved one reaches the pipeline as
    the literal string."""
    monkeypatch.setattr(infer.hydra.utils, "instantiate", lambda cfg, **kwargs: DummyDataModule(**cfg))

    cfg = OmegaConf.create(
        {
            "data": {
                "datamodule": {
                    "_target_": "tests.test_infer.DummyDataModule",
                    "artifact_bank_dir": "${paths.artifact_bank_dir}",
                    "artifact_taxonomy_csv": "${paths.root_dir}/data/artifact/artifact_taxonomy_by_image.csv",
                }
            },
            "infer": {"runtime": {"batch_size_override": None}},
            "paths": {"root_dir": "/tmp/fake-root", "artifact_bank_dir": "/tmp/fake-bank"},
        }
    )

    datamodule = infer._instantiate_datamodule(cfg)

    assert datamodule.artifact_bank_dir == "/tmp/fake-bank"
    assert (
        datamodule.artifact_taxonomy_csv
        == "/tmp/fake-root/data/artifact/artifact_taxonomy_by_image.csv"
    )


def test_extract_ckpt_run_id_finds_the_project_timestamp():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/"
        "runs/2026-08-20_14-03-11/checkpoints/last.ckpt"
    )
    assert infer._extract_ckpt_run_id(ckpt_path) == "2026-08-20_14-03-11"


def test_extract_ckpt_run_id_falls_back_when_no_timestamp_present():
    run_id = infer._extract_ckpt_run_id("/some/renamed/checkpoint.ckpt")
    assert infer._RUN_ID_PATTERN.fullmatch(run_id)


def test_derive_default_run_name_uses_ckpt_run_id_and_dataset():
    ckpt_path = (
        "/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_acevedo/"
        "runs/2026-08-20_14-03-11/checkpoints/best.ckpt"
    )
    run_name = infer.derive_default_run_name(ckpt_path, net_name="baseline_classifier", data_name="acevedo")
    assert run_name == "baseline_classifier_acevedo/2026-08-20_14-03-11/acevedo"


def test_derive_default_run_name_fallback_shape():
    run_name = infer.derive_default_run_name(
        "/tmp/wandb-artifacts/model-v3/model.ckpt", net_name="deep_ensemble", data_name="tang"
    )
    assert re.fullmatch(rf"deep_ensemble/{infer._RUN_ID_PATTERN.pattern}/tang", run_name)


def test_derive_default_run_name_tags_mc_dropout():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/"
        "runs/2026-08-20_14-03-11/checkpoints/best.ckpt"
    )
    run_name = infer.derive_default_run_name(
        ckpt_path, net_name="sngp_classifier", data_name="acevedo", use_mc_dropout=True
    )
    assert run_name == "mc_sngp_classifier_acevedo/2026-08-20_14-03-11/acevedo"


def test_extract_netname_dataset_parses_runs_segment():
    ckpt_path = (
        "/data1/maheswararao/experiments/uncertaity-aware-ml/train/baseline_classifier_acevedo/"
        "runs/2026-08-25_14-31-36/checkpoints/best.ckpt"
    )
    assert infer._extract_netname_dataset(ckpt_path, fallback="model") == "baseline_classifier_acevedo"


def test_extract_netname_dataset_parses_ensemble_members_segment():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/"
        "ensemble_members/2026-08-20_14-03-11/checkpoints/ensemble.ckpt"
    )
    assert infer._extract_netname_dataset(ckpt_path, fallback="model") == "sngp_classifier_acevedo"


def test_extract_netname_dataset_falls_back_when_no_convention_segment():
    netname_dataset = infer._extract_netname_dataset("/tmp/wandb-artifacts/model-v3/model.ckpt", fallback="model")
    assert netname_dataset == "model"


def test_derive_default_run_name_deep_ensemble_assembled_path():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/sngp_classifier_acevedo/"
        "ensemble_members/2026-08-20_14-03-11/checkpoints/ensemble.ckpt"
    )
    run_name = infer.derive_default_run_name(
        ckpt_path, net_name="deep_ensemble", data_name="acevedo", is_deep_ensemble=True
    )
    assert run_name == "deep_ensemble_sngp_classifier_acevedo/2026-08-20_14-03-11/acevedo"


def test_derive_default_run_name_deep_ensemble_sequential_path_no_doubled_prefix():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/deep_ensemble_classifier_acevedo/"
        "runs/2026-08-20_14-03-11/checkpoints/best.ckpt"
    )
    run_name = infer.derive_default_run_name(
        ckpt_path, net_name="deep_ensemble", data_name="acevedo", is_deep_ensemble=True
    )
    assert run_name == "deep_ensemble_classifier_acevedo/2026-08-20_14-03-11/acevedo"
    assert "deep_ensemble_deep_ensemble" not in run_name


def test_derive_default_run_name_deep_ensemble_precedes_mc_dropout():
    ckpt_path = (
        "/data1/experiments/uncertaity-aware-ml/train/baseline_classifier_tang/"
        "ensemble_members/2026-08-20_14-03-11/checkpoints/ensemble.ckpt"
    )
    run_name = infer.derive_default_run_name(
        ckpt_path,
        net_name="deep_ensemble",
        data_name="tang",
        use_mc_dropout=True,
        is_deep_ensemble=True,
    )
    assert run_name == "deep_ensemble_baseline_classifier_tang/2026-08-20_14-03-11/tang"
    assert "mc_" not in run_name


# --- src/inference/records.py: the shared write-path both runners go through -------


class _FakeNet(torch.nn.Module):
    """Returns a canned `ModelOutput`, so record-building can be exercised without a
    real checkpoint or dataset."""

    def __init__(self, output: ModelOutput) -> None:
        super().__init__()
        self._output = output

    def forward(self, x: torch.Tensor) -> ModelOutput:
        return self._output


def _batch_outputs(logits: torch.Tensor, **kwargs) -> records.BatchOutputs:
    return records.BatchOutputs(logits=logits, probs=torch.softmax(logits, dim=1), **kwargs)


def test_extract_model_outputs_reads_every_model_output_field():
    logits = torch.tensor([[1.0, 2.0], [0.5, -0.5]])
    model = _FakeNet(
        ModelOutput(
            logits=logits,
            raw_logits=logits * 2,
            variance=torch.tensor([[0.25], [0.75]]),
            member_logits=torch.stack([logits, logits + 1.0]),
        )
    )

    out = records.extract_model_outputs(model, torch.zeros(2, 3), use_mc_dropout=False, mc_passes=5)

    assert torch.equal(out.logits, logits)
    assert torch.allclose(out.probs, torch.softmax(logits, dim=1))
    assert torch.equal(out.raw_logits, logits * 2)
    assert out.uncertainty.shape == (2, 1)
    assert out.member_logits.shape == (2, 2, 2)


def test_extract_model_outputs_handles_legacy_bare_tuple_nets():
    """Pre-`ModelOutput` nets returned a bare (mean_field_logits, raw_logits, pred_var)."""

    class LegacyNet(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]]), torch.tensor([[0.5]])

    out = records.extract_model_outputs(LegacyNet(), torch.zeros(1, 3), use_mc_dropout=False, mc_passes=5)

    assert torch.equal(out.logits, torch.tensor([[1.0, 0.0]]))
    assert torch.equal(out.raw_logits, torch.tensor([[2.0, 0.0]]))
    assert torch.equal(out.uncertainty, torch.tensor([[0.5]]))
    assert out.member_logits is None


def test_extract_model_outputs_handles_bare_tensor_nets():
    class BareNet(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[1.0, 0.0]])

    out = records.extract_model_outputs(BareNet(), torch.zeros(1, 3), use_mc_dropout=False, mc_passes=5)

    assert out.uncertainty is None
    assert out.raw_logits is None
    assert torch.allclose(out.probs.sum(dim=1), torch.ones(1))


def test_build_records_emits_one_row_per_sample_with_expected_columns():
    outputs = _batch_outputs(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))

    rows = records.build_records(
        image_ids=["a", "b"],
        fold=["test", "test"],
        targets=torch.tensor([0, 1]),
        preds=torch.tensor([0, 1]),
        outputs=outputs,
        stream_name="default",
    )

    assert len(rows) == 2
    assert list(rows[0]) == [
        "image_id",
        "fold",
        "target",
        "prediction",
        "confidence",
        "class_logits",
        "class_probs",
        "stream",
    ]
    assert rows[0]["image_id"] == "a"
    assert rows[1]["target"] == 1
    # class_probs is JSON, not a repr'd Python list -- downstream readers json.loads it.
    assert json.loads(rows[0]["class_probs"]) == pytest.approx(outputs.probs[0].tolist())


def test_build_records_omits_uncertainty_when_the_net_has_none():
    rows = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=_batch_outputs(torch.tensor([[1.0, 0.0]])),
        stream_name="default",
    )
    assert "uncertainty" not in rows[0]


def test_build_records_includes_uncertainty_when_present():
    rows = records.build_records(
        image_ids=["a", "b"],
        fold=["test", "test"],
        targets=torch.tensor([0, 1]),
        preds=torch.tensor([0, 1]),
        outputs=_batch_outputs(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), uncertainty=torch.tensor([[0.25], [0.75]])
        ),
        stream_name="default",
    )
    assert [r["uncertainty"] for r in rows] == pytest.approx([0.25, 0.75])


def test_build_records_carries_the_stream_name():
    rows = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=_batch_outputs(torch.tensor([[1.0, 0.0]])),
        stream_name="artifact",
    )
    assert rows[0]["stream"] == "artifact"


def test_write_outputs_writes_both_files(tmp_path):
    rows = [{"image_id": "a", "target": 0, "prediction": 0}]

    records.write_outputs(
        output_root=tmp_path,
        records=rows,
        metrics={"acc": 1.0},
        save_cfg={"save_csv": True, "save_metrics_json": True},
    )

    df = pd.read_csv(tmp_path / "predictions.csv")
    assert df.to_dict("records") == rows
    assert json.loads((tmp_path / "metrics.json").read_text()) == {"acc": 1.0}


def test_write_outputs_honors_the_save_flags(tmp_path):
    records.write_outputs(
        output_root=tmp_path,
        records=[{"image_id": "a"}],
        metrics={"acc": 1.0},
        save_cfg={"save_csv": False, "save_metrics_json": False},
    )

    assert not (tmp_path / "predictions.csv").exists()
    assert not (tmp_path / "metrics.json").exists()


def test_write_outputs_accepts_a_dictconfig_save_section(tmp_path):
    """`_finalize` passes `cfg.infer.save` straight through, so a DictConfig has to work."""
    records.write_outputs(
        output_root=tmp_path,
        records=[{"image_id": "a"}],
        metrics={},
        save_cfg=OmegaConf.create({"save_csv": True, "save_metrics_json": True}),
    )
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "metrics.json").exists()


# --- class_logits / raw_logits persistence, and the uncertainty reduction ---------


def test_build_records_always_writes_class_logits():
    """Softmax is shift-invariant, so logits are unrecoverable from class_probs --
    they have to be persisted at write time or not at all."""
    logits = torch.tensor([[2.0, 0.0], [-1.0, 3.0]])
    rows = records.build_records(
        image_ids=["a", "b"],
        fold=["test", "test"],
        targets=torch.tensor([0, 1]),
        preds=torch.tensor([0, 1]),
        outputs=_batch_outputs(logits),
        stream_name="default",
    )

    assert json.loads(rows[0]["class_logits"]) == pytest.approx([2.0, 0.0])
    assert json.loads(rows[1]["class_logits"]) == pytest.approx([-1.0, 3.0])


def test_class_logits_precedes_class_probs_in_column_order():
    rows = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=_batch_outputs(torch.tensor([[1.0, 0.0]])),
        stream_name="default",
    )
    columns = list(rows[0])
    assert columns.index("class_logits") == columns.index("class_probs") - 1


def test_class_logits_softmax_matches_class_probs_for_a_plain_forward():
    """The consistency guarantee downstream consumers may rely on -- for every runner
    path except MC-Dropout (see the paired test below)."""
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    rows = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=_batch_outputs(logits),
        stream_name="default",
    )

    recovered = torch.softmax(torch.tensor(json.loads(rows[0]["class_logits"])), dim=0)
    assert recovered.tolist() == pytest.approx(json.loads(rows[0]["class_probs"]))


def test_class_logits_softmax_deliberately_differs_from_class_probs_under_mc_dropout():
    """`mc_predict` returns mean-of-logits alongside mean-of-softmax, and those are not
    the same thing. Pinned so nobody "fixes" it into a consistency bug report."""

    class _McNet(torch.nn.Module):
        def mc_predict(self, x, T, return_std, apply_softmax):
            # Deliberately asymmetric: symmetric passes make the two estimators coincide.
            per_pass = torch.tensor([[[4.0, 0.0]], [[0.0, 0.0]]])  # [T=2, B=1, C=2]
            mean_logits = per_pass.mean(0)
            mean_probs = torch.softmax(per_pass, dim=-1).mean(0)
            return mean_logits, mean_probs, per_pass.std(0, unbiased=False)

    out = records.extract_model_outputs(_McNet(), torch.zeros(1, 3), use_mc_dropout=True, mc_passes=2)
    rows = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=out,
        stream_name="default",
    )

    recovered = torch.softmax(torch.tensor(json.loads(rows[0]["class_logits"])), dim=0)
    assert recovered.tolist() != pytest.approx(json.loads(rows[0]["class_probs"]))
    # softmax(mean logits [2, 0]) vs mean(softmax([4,0]), softmax([0,0])).
    assert recovered.tolist() == pytest.approx([0.8808, 0.1192], abs=1e-4)
    assert json.loads(rows[0]["class_probs"]) == pytest.approx([0.7411, 0.2589], abs=1e-4)


def test_build_records_writes_raw_logits_only_when_the_net_emits_them():
    with_raw = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=_batch_outputs(torch.tensor([[1.0, 0.0]]), raw_logits=torch.tensor([[3.0, 1.0]])),
        stream_name="default",
    )
    without_raw = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=_batch_outputs(torch.tensor([[1.0, 0.0]])),
        stream_name="default",
    )

    assert json.loads(with_raw[0]["raw_logits"]) == pytest.approx([3.0, 1.0])
    assert "raw_logits" not in without_raw[0]


def test_per_class_uncertainty_is_reduced_to_one_value_per_sample():
    """mc_predict's std is [B, C]; the record column is one scalar per sample."""
    per_class = torch.tensor([[0.2, 0.4], [1.0, 3.0]])
    assert records._reduce_uncertainty(per_class).shape == (2, 1)
    assert records._reduce_uncertainty(per_class).flatten().tolist() == pytest.approx([0.3, 2.0])


def test_already_scalar_uncertainty_passes_through_unchanged():
    """SNGP / DeepEnsemble emit [B, 1] and must not be touched."""
    scalar = torch.tensor([[0.25], [0.75]])
    assert torch.equal(records._reduce_uncertainty(scalar), scalar)
    assert records._reduce_uncertainty(None) is None


def test_build_records_rejects_a_misaligned_uncertainty_tensor():
    """The old code flattened [B, C] and silently took the first B values, so every
    mc_* run's uncertainty column was misaligned. Now it fails loudly."""
    outputs = records.BatchOutputs(
        logits=torch.zeros(2, 2),
        probs=torch.full((2, 2), 0.5),
        uncertainty=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),  # unreduced [B, C]
    )

    with pytest.raises(ValueError, match="expected one per sample"):
        records.build_records(
            image_ids=["a", "b"],
            fold=["test", "test"],
            targets=torch.tensor([0, 1]),
            preds=torch.tensor([0, 1]),
            outputs=outputs,
            stream_name="default",
        )
