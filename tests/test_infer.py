import json
import re
from pathlib import Path

import pandas as pd
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from src.inference import infer, records
from src.inference.infer_artifact import ArtifactInferenceRunner
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


def test_build_records_omits_artifact_axis_fields_by_default():
    rows = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=_batch_outputs(torch.tensor([[1.0, 0.0]])),
        stream_name="real",
    )
    assert "count" not in rows[0]
    assert "severity" not in rows[0]
    assert "percent_pixels_affected" not in rows[0]
    assert "global_degradations" not in rows[0]
    assert "geometric_degradations" not in rows[0]


def test_build_records_includes_artifact_axis_fields_when_passed():
    rows = records.build_records(
        image_ids=["a", "b"],
        fold=["test", "test"],
        targets=torch.tensor([0, 1]),
        preds=torch.tensor([0, 1]),
        outputs=_batch_outputs(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        stream_name="artifact",
        count=1,
        severity=3,
        percent_pixels_affected=torch.tensor([2.5, 0.0]),
        global_degradations=['["hed_stain_shift"]', "[]"],
        geometric_degradations=["[]", '["elastic_deformation"]'],
    )
    assert [r["count"] for r in rows] == [1, 1]
    assert [r["severity"] for r in rows] == [3, 3]
    assert [r["percent_pixels_affected"] for r in rows] == pytest.approx([2.5, 0.0])
    assert [r["global_degradations"] for r in rows] == ['["hed_stain_shift"]', "[]"]
    assert [r["geometric_degradations"] for r in rows] == ["[]", '["elastic_deformation"]']


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


# --- run.json provenance sidecar ---------------------------------------------------


def test_write_outputs_writes_run_json_with_provenance_and_derived_fields(tmp_path):
    rows = [{"image_id": "a", "class_logits": "[1.0, 2.0]", "class_probs": "[0.1, 0.9]"}]
    provenance = {
        "predictions_csv_schema": 2,
        "ckpt_path": "/data1/x/train/baseline_classifier_acevedo/runs/2026-08-20_14-03-11/checkpoints/best.ckpt",
        "member_source": None,
        "num_members": None,
    }

    records.write_outputs(
        output_root=tmp_path,
        records=rows,
        metrics={"acc": 1.0},
        save_cfg={"save_csv": True, "save_metrics_json": True, "save_run_json": True},
        provenance=provenance,
    )

    run_json = json.loads((tmp_path / "run.json").read_text())
    assert run_json["ckpt_path"] == provenance["ckpt_path"]
    assert run_json["predictions_csv_schema"] == 2
    assert run_json["n_rows"] == 1
    assert run_json["columns"] == ["image_id", "class_logits", "class_probs"]
    assert "written_at" in run_json


def test_write_outputs_omits_run_json_when_the_flag_is_false(tmp_path):
    records.write_outputs(
        output_root=tmp_path,
        records=[{"image_id": "a"}],
        metrics={},
        save_cfg={"save_csv": True, "save_metrics_json": True, "save_run_json": False},
        provenance={"ckpt_path": "x"},
    )
    assert not (tmp_path / "run.json").exists()


def test_write_outputs_omits_run_json_when_provenance_is_none(tmp_path):
    records.write_outputs(
        output_root=tmp_path,
        records=[{"image_id": "a"}],
        metrics={},
        save_cfg={"save_csv": True, "save_metrics_json": True, "save_run_json": True},
        provenance=None,
    )
    assert not (tmp_path / "run.json").exists()


def test_write_outputs_run_json_columns_empty_when_no_records(tmp_path):
    records.write_outputs(
        output_root=tmp_path,
        records=[],
        metrics={},
        save_cfg={"save_csv": True, "save_metrics_json": True, "save_run_json": True},
        provenance={"ckpt_path": "x"},
    )
    run_json = json.loads((tmp_path / "run.json").read_text())
    assert run_json["columns"] == []
    assert run_json["n_rows"] == 0


# --- member_logits: capture_members / save_member_logits opt-in ---------------------


def test_extract_model_outputs_ensemble_member_logits_is_free_regardless_of_capture_members():
    """DeepEnsemble already returns member_logits on ModelOutput -- no extra compute,
    so it should show up whether or not capture_members is requested."""
    logits = torch.tensor([[1.0, 2.0]])
    member_logits = torch.stack([logits, logits + 1.0, logits + 2.0])  # [M=3, B=1, C=2]
    model = _FakeNet(ModelOutput(logits=logits, member_logits=member_logits))

    for capture_members in (False, True):
        out = records.extract_model_outputs(
            model, torch.zeros(1, 3), use_mc_dropout=False, mc_passes=5, capture_members=capture_members
        )
        assert out.member_logits.shape == (3, 1, 2)


def test_extract_model_outputs_mc_dropout_without_capture_members_omits_member_logits():
    class _McNet(torch.nn.Module):
        def mc_predict(self, x, T, return_std, apply_softmax):
            per_pass = torch.randn(T, x.shape[0], 2)
            return per_pass.mean(0), torch.softmax(per_pass, dim=-1).mean(0), per_pass.std(0, unbiased=False)

    out = records.extract_model_outputs(
        _McNet(), torch.zeros(2, 3), use_mc_dropout=True, mc_passes=4, capture_members=False
    )
    assert out.member_logits is None


def test_extract_model_outputs_mc_dropout_with_capture_members_populates_member_logits():
    class _McNet(torch.nn.Module):
        def mc_predict(self, x, T, return_std, apply_softmax):
            raise AssertionError("capture_members=True must call mc_forward_samples, not mc_predict")

        def mc_forward_samples(self, x, T):
            torch.manual_seed(0)
            return torch.randn(T, x.shape[0], 2)

    out = records.extract_model_outputs(
        _McNet(), torch.zeros(2, 3), use_mc_dropout=True, mc_passes=4, capture_members=True
    )

    assert out.member_logits.shape == (4, 2, 2)
    torch.manual_seed(0)
    expected_stack = torch.randn(4, 2, 2)
    assert torch.allclose(out.logits, expected_stack.mean(0))
    assert torch.allclose(out.probs, torch.softmax(expected_stack, dim=-1).mean(0))
    assert torch.allclose(out.uncertainty, expected_stack.std(0, unbiased=False).mean(dim=1, keepdim=True))


def test_build_records_writes_member_logits_only_when_flag_and_data_both_present():
    member_logits = torch.stack(
        [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[2.0, 0.0], [0.0, 2.0]])]
    )  # [M=2, B=2, C=2]
    outputs = _batch_outputs(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), member_logits=member_logits)

    with_flag = records.build_records(
        image_ids=["a", "b"],
        fold=["test", "test"],
        targets=torch.tensor([0, 1]),
        preds=torch.tensor([0, 1]),
        outputs=outputs,
        stream_name="default",
        save_member_logits=True,
    )
    without_flag = records.build_records(
        image_ids=["a", "b"],
        fold=["test", "test"],
        targets=torch.tensor([0, 1]),
        preds=torch.tensor([0, 1]),
        outputs=outputs,
        stream_name="default",
        save_member_logits=False,
    )

    assert "member_logits" not in without_flag[0]
    # member_logits_cpu[idx] is that sample's [M, C]: sample 0 across both members.
    assert json.loads(with_flag[0]["member_logits"]) == [[1.0, 0.0], [2.0, 0.0]]
    assert json.loads(with_flag[1]["member_logits"]) == [[0.0, 1.0], [0.0, 2.0]]


def test_build_records_omits_member_logits_when_flag_set_but_no_member_data():
    rows = records.build_records(
        image_ids=["a"],
        fold=["test"],
        targets=torch.tensor([0]),
        preds=torch.tensor([0]),
        outputs=_batch_outputs(torch.tensor([[1.0, 0.0]])),
        stream_name="default",
        save_member_logits=True,
    )
    assert "member_logits" not in rows[0]


def _artifact_cfg(tmp_path: Path, streams: list, artifact_count=None, artifact_severity=None) -> DictConfig:
    return OmegaConf.create(
        {
            "save_path": str(tmp_path),
            "infer": {
                "runtime": {"device": "cpu", "use_mc_dropout": False, "mc_passes": 1},
                "metrics": {"enabled": False, "items": []},
                "save": {
                    "run_name": "run",
                    "streams": streams,
                    "save_csv": True,
                    "save_metrics_json": True,
                    "save_run_json": False,
                    "save_member_logits": False,
                    "save_images": False,
                    "max_images_to_save": 64,
                },
            },
            "data": {"datamodule": {"artifact_count": artifact_count, "artifact_severity": artifact_severity}},
        }
    )


def _artifact_batch() -> dict:
    return {
        "image_id": ["a", "b"],
        "fold": ["test", "test"],
        "target": torch.tensor([0, 1]),
        "real_image": torch.zeros(2, 3, 4, 4),
        "artifact_simulated_image": torch.ones(2, 3, 4, 4),
        "percent_pixels_affected": torch.tensor([2.5, 0.0]),
        "global_degradations": ['["hed_stain_shift"]', "[]"],
        "geometric_degradations": ["[]", "[]"],
    }


def test_artifact_runner_narrowed_to_artifact_stream_skips_real_forward_pass_and_rows(tmp_path):
    """`infer.save.streams=[artifact]` -- the sweep-run shape -- must never touch the real
    stream: no forward pass, no records, and count/severity/percent_pixels_affected/
    global_degradations land only on the rows it does write."""
    model = _FakeNet(ModelOutput(logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
    cfg = _artifact_cfg(tmp_path, streams=["artifact"], artifact_count=1, artifact_severity=None)
    runner = ArtifactInferenceRunner(model=model, dataloader=[_artifact_batch()], cfg=cfg)

    runner.run()

    streams_seen = {r["stream"] for r in runner._records}
    assert streams_seen == {"artifact"}
    assert all(r["count"] == 1 for r in runner._records)
    assert [r["percent_pixels_affected"] for r in runner._records] == pytest.approx([2.5, 0.0])
    assert [r["global_degradations"] for r in runner._records] == ['["hed_stain_shift"]', "[]"]


def test_artifact_runner_default_streams_writes_both(tmp_path):
    model = _FakeNet(ModelOutput(logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]])))
    cfg = _artifact_cfg(tmp_path, streams=["real", "artifact"])
    runner = ArtifactInferenceRunner(model=model, dataloader=[_artifact_batch()], cfg=cfg)

    runner.run()

    streams_seen = {r["stream"] for r in runner._records}
    assert streams_seen == {"real", "artifact"}
    # Real-stream rows never carry the artifact-only fields.
    real_rows = [r for r in runner._records if r["stream"] == "real"]
    assert all(
        "count" not in r and "percent_pixels_affected" not in r and "global_degradations" not in r
        for r in real_rows
    )


def test_artifact_runner_rejects_invalid_stream_name(tmp_path):
    model = _FakeNet(ModelOutput(logits=torch.tensor([[1.0, 0.0]])))
    cfg = _artifact_cfg(tmp_path, streams=["bogus"])
    with pytest.raises(ValueError, match="real.*artifact"):
        ArtifactInferenceRunner(model=model, dataloader=[_artifact_batch()], cfg=cfg)
