"""Tests for src/metrics/manifest.py.

Split into two tiers per the project's testing policy: pure-logic checks against a
small synthetic manifest (no disk, no env vars, run anywhere) and a real-disk check
against the actual checked-in configs/runs/infer_manifest.yaml (skipped when
EXPERIMENTS_HOME/PROJECT_NAME aren't set, i.e. off this machine/off CI).
"""

import os

import pytest
from omegaconf import OmegaConf

from src.metrics.manifest import (
    DEFAULT_MANIFEST_PATH,
    RunEntry,
    check_master_docs_are_current,
    load_manifest,
    render_ckpt_master_doc,
    render_infer_master_doc,
    resolve_default_infer_root,
    validate_manifest,
)

_HAS_REAL_ENV = bool(os.environ.get("EXPERIMENTS_HOME")) and bool(os.environ.get("PROJECT_NAME"))


def _write_synthetic_manifest(path):
    OmegaConf.save(
        OmegaConf.create(
            {
                "schema_version": 1,
                "infer_root": "will-be-overridden",
                "runs": [
                    {
                        "label": "baseline_a",
                        "method": "baseline",
                        "train_dataset": "a",
                        "is_ensemble": False,
                        "uses_mc_dropout": False,
                        "run_dir": "baseline_classifier_a/2026-01-01_00-00-00",
                        "ckpt": "train/baseline_classifier_a/runs/2026-01-01_00-00-00/checkpoints/best.ckpt",
                        "id_dataset": "a",
                        "eval_datasets": ["a", "b"],
                        "paired_streams": [],
                    },
                    {
                        "label": "mc_baseline_a",
                        "method": "mc_dropout",
                        "train_dataset": "a",
                        "is_ensemble": False,
                        "uses_mc_dropout": True,
                        "run_dir": "mc_baseline_classifier_a/2026-01-01_00-00-00",
                        "ckpt": "train/baseline_classifier_a/runs/2026-01-01_00-00-00/checkpoints/best.ckpt",
                        "id_dataset": "a",
                        "eval_datasets": ["a", "b"],
                        "paired_streams": [],
                    },
                    {
                        "label": "baseline_c",
                        "method": "baseline",
                        "train_dataset": "c",
                        "is_ensemble": False,
                        "uses_mc_dropout": False,
                        "run_dir": "baseline_classifier_c/2026-02-02_00-00-00",
                        "ckpt": "train/baseline_classifier_c/runs/2026-02-02_00-00-00/checkpoints/best.ckpt",
                        "id_dataset": "c",
                        "eval_datasets": ["c"],
                        "paired_streams": ["c_artifact"],
                    },
                ],
            }
        ),
        path,
    )
    return path


# --- load_manifest ---------------------------------------------------------------


def test_load_manifest_infer_root_override_bypasses_env_resolution(tmp_path):
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))
    assert str(manifest.infer_root) == str(tmp_path / "infer")
    assert len(manifest.runs) == 3


def test_load_manifest_parses_run_entry_fields(tmp_path):
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))

    run = manifest.get("baseline_a")
    assert isinstance(run, RunEntry)
    assert run.method == "baseline"
    assert run.eval_datasets == ("a", "b")
    assert run.paired_streams == ()

    with_paired = manifest.get("baseline_c")
    assert with_paired.paired_streams == ("c_artifact",)


def test_manifest_get_raises_key_error_listing_labels(tmp_path):
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))
    with pytest.raises(KeyError, match="baseline_a"):
        manifest.get("not_a_real_label")


def test_resolve_default_infer_root_raises_system_exit_when_env_unset(monkeypatch):
    monkeypatch.delenv("EXPERIMENTS_HOME", raising=False)
    monkeypatch.delenv("PROJECT_NAME", raising=False)
    with pytest.raises(SystemExit, match="--infer-root"):
        resolve_default_infer_root()


def test_resolve_default_infer_root_matches_paths_convention(monkeypatch):
    monkeypatch.setenv("EXPERIMENTS_HOME", "/tmp/exp")
    monkeypatch.setenv("PROJECT_NAME", "proj")
    assert resolve_default_infer_root() == str(__import__("pathlib").Path("/tmp/exp/proj/infer"))


# --- validate_manifest: pure-logic checks (no disk) -------------------------------


def test_validate_manifest_clean_synthetic_manifest_has_no_problems(tmp_path):
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))
    assert validate_manifest(manifest, check_disk=False) == []


def test_validate_manifest_flags_duplicate_labels(tmp_path):
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))
    dup = manifest.runs[0]
    manifest = manifest.__class__(manifest.schema_version, manifest.infer_root, manifest.runs + (dup,))

    problems = validate_manifest(manifest, check_disk=False)
    assert any("duplicate label" in p for p in problems)


def test_validate_manifest_flags_id_dataset_not_in_eval_datasets(tmp_path):
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))
    bad_run = manifest.runs[0].__class__(
        **{**manifest.runs[0].__dict__, "id_dataset": "not_in_eval_datasets"}
    )
    manifest = manifest.__class__(manifest.schema_version, manifest.infer_root, (bad_run,) + manifest.runs[1:])

    problems = validate_manifest(manifest, check_disk=False)
    assert any("not in its own eval_datasets" in p for p in problems)


def test_validate_manifest_flags_run_dir_ckpt_run_id_mismatch(tmp_path):
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))
    bad_run = manifest.runs[0].__class__(
        **{**manifest.runs[0].__dict__, "run_dir": "baseline_classifier_a/2099-01-01_00-00-00"}
    )
    manifest = manifest.__class__(manifest.schema_version, manifest.infer_root, (bad_run,) + manifest.runs[1:])

    problems = validate_manifest(manifest, check_disk=False)
    assert any("does not match ckpt run-id" in p for p in problems)


# --- validate_manifest: disk checks (synthetic tmp_path tree) ---------------------


def _touch_predictions_csv(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "predictions.csv").write_text("image_id\n")


def test_validate_manifest_disk_checks_pass_for_a_fully_populated_tree(tmp_path):
    infer_root = tmp_path / "infer"
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(infer_root))

    for run in manifest.runs:
        for leaf in (*run.eval_datasets, *run.paired_streams):
            _touch_predictions_csv(infer_root / run.run_dir / leaf)

    assert validate_manifest(manifest, check_disk=True) == []


def test_validate_manifest_flags_a_missing_run_dir(tmp_path):
    infer_root = tmp_path / "infer"
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(infer_root))
    # Populate everything except the first run's dir.
    for run in manifest.runs[1:]:
        for leaf in (*run.eval_datasets, *run.paired_streams):
            _touch_predictions_csv(infer_root / run.run_dir / leaf)

    problems = validate_manifest(manifest, check_disk=True)
    assert any("run_dir does not exist" in p for p in problems)


def test_validate_manifest_flags_a_missing_predictions_csv(tmp_path):
    infer_root = tmp_path / "infer"
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(infer_root))
    for run in manifest.runs:
        for leaf in (*run.eval_datasets, *run.paired_streams):
            _touch_predictions_csv(infer_root / run.run_dir / leaf)
    # Delete one predictions.csv after the fact.
    (infer_root / manifest.runs[0].run_dir / "a" / "predictions.csv").unlink()

    problems = validate_manifest(manifest, check_disk=True)
    assert any("no predictions.csv" in p for p in problems)


def test_validate_manifest_flags_an_unclaimed_leaf_on_disk(tmp_path):
    """Catches drift the other direction: inference produced a dataset leaf the
    manifest doesn't know about."""
    infer_root = tmp_path / "infer"
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(infer_root))
    for run in manifest.runs:
        for leaf in (*run.eval_datasets, *run.paired_streams):
            _touch_predictions_csv(infer_root / run.run_dir / leaf)
    _touch_predictions_csv(infer_root / manifest.runs[0].run_dir / "surprise_dataset")

    problems = validate_manifest(manifest, check_disk=True)
    assert any("not claimed" in p for p in problems)


def test_validate_manifest_ignores_an_images_subdir(tmp_path):
    infer_root = tmp_path / "infer"
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(infer_root))
    for run in manifest.runs:
        for leaf in (*run.eval_datasets, *run.paired_streams):
            _touch_predictions_csv(infer_root / run.run_dir / leaf)
    (infer_root / manifest.runs[0].run_dir / "images").mkdir(parents=True)

    assert validate_manifest(manifest, check_disk=True) == []


# --- doc rendering -----------------------------------------------------------------


def test_render_infer_master_doc_groups_by_id_dataset_with_separators(tmp_path):
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))

    doc = render_infer_master_doc(manifest)
    assert "baseline_a:" in doc
    assert "mc_baseline_a:" in doc
    assert "baseline_c:" in doc
    assert doc.count("---") == 1  # two id_dataset groups -> one separator
    assert "do not edit by hand" in doc


def test_render_ckpt_master_doc_omits_mc_dropout_entries(tmp_path):
    """mc_baseline_a reuses baseline_a's checkpoint -- the ckpt doc should have no
    separate entry for it, matching the original hand-written doc's own shape."""
    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))

    doc = render_ckpt_master_doc(manifest)
    assert "baseline_a:" in doc
    assert "mc_baseline_a" not in doc


def test_check_master_docs_are_current_flags_missing_files(tmp_path, monkeypatch):
    import src.metrics.manifest as manifest_module

    monkeypatch.setattr(manifest_module, "MASTER_INFER_DOC_PATH", tmp_path / "nope_infer.md")
    monkeypatch.setattr(manifest_module, "MASTER_CKPT_DOC_PATH", tmp_path / "nope_ckpt.md")

    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))

    problems = check_master_docs_are_current(manifest)
    assert len(problems) == 2


def test_check_master_docs_are_current_passes_when_freshly_written(tmp_path, monkeypatch):
    import src.metrics.manifest as manifest_module

    infer_doc = tmp_path / "infer.md"
    ckpt_doc = tmp_path / "ckpt.md"
    monkeypatch.setattr(manifest_module, "MASTER_INFER_DOC_PATH", infer_doc)
    monkeypatch.setattr(manifest_module, "MASTER_CKPT_DOC_PATH", ckpt_doc)

    path = _write_synthetic_manifest(tmp_path / "manifest.yaml")
    manifest = load_manifest(path, infer_root=str(tmp_path / "infer"))

    infer_doc.write_text(render_infer_master_doc(manifest))
    ckpt_doc.write_text(render_ckpt_master_doc(manifest))

    assert check_master_docs_are_current(manifest) == []


# --- the real, checked-in manifest -------------------------------------------------


def test_real_manifest_is_internally_consistent_with_no_disk_access():
    """Runs anywhere -- no EXPERIMENTS_HOME/PROJECT_NAME needed, since infer_root is
    overridden to a placeholder that's never touched by check_disk=False."""
    manifest = load_manifest(DEFAULT_MANIFEST_PATH, infer_root="/placeholder")
    assert validate_manifest(manifest, check_disk=False) == []


@pytest.mark.skipif(not _HAS_REAL_ENV, reason="requires EXPERIMENTS_HOME/PROJECT_NAME and the real infer/ tree")
def test_real_manifest_matches_disk_and_generated_docs_are_current():
    manifest = load_manifest(DEFAULT_MANIFEST_PATH, infer_root=resolve_default_infer_root())
    assert validate_manifest(manifest, check_disk=True) == []
    assert check_master_docs_are_current(manifest) == []
