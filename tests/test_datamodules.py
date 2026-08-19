"""test_datamodules.py in tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.checkpointing.io import read_meta
from src.data.artifact_image_datamodule import ArtifactHFDataset, ArtifactImageDataModule
from src.data.classification_image_datamodule import ClassificationImageDataModule
from src.data.components.hf_dataset import HFDataset
from src.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


def test_classification_datamodule_setup_stage_test_builds_test_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct stage='test' setup should populate the test dataset for standalone inference."""

    def fake_load_dataset(_dataset_name: str):
        return {
            "train": [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 0, "image_id": "train-0"}],
            "validation": [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 1, "image_id": "val-0"}],
            "test": [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 2, "image_id": "test-0"}],
        }

    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        fake_load_dataset,
    )

    dm = ClassificationImageDataModule(dataset_name="dummy", batch_size=1, num_workers=0, pin_memory=False)
    dm.setup(stage="test")

    assert dm.data_test is not None
    batch = next(iter(dm.test_dataloader()))
    image_ids, images, labels, fold = batch
    assert len(image_ids) == 1
    assert images.shape[0] == 1
    assert labels.shape[0] == 1
    assert fold[0] == "test"


def test_classification_datamodule_rejects_length_mismatched_class_to_idx() -> None:
    """class_to_idx length must match num_classes at construction time, before any
    data is loaded."""
    with pytest.raises(AssertionError, match="class_to_idx"):
        ClassificationImageDataModule(
            dataset_name="dummy",
            num_classes=3,
            class_to_idx={"a": 0, "b": 1},
            batch_size=1,
            num_workers=0,
            pin_memory=False,
        )


def _fake_load_dataset_with_classes(classes_to_idx: dict):
    def fake_load_dataset(_dataset_name: str):
        return {
            "train": [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 0, "image_id": "train-0"}],
            "validation": [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 1, "image_id": "val-0"}],
            "test": [
                {
                    "image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "label": 0,
                    "image_id": "test-0",
                    "classes_to_idx": str(classes_to_idx),
                }
            ],
        }

    return fake_load_dataset


def test_classification_datamodule_accepts_matching_class_to_idx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured class_to_idx that matches the dataset's own mapping should be
    accepted and exposed on the datamodule."""
    fake_classes = {"cat": 0, "dog": 1}
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        _fake_load_dataset_with_classes(fake_classes),
    )

    dm = ClassificationImageDataModule(
        dataset_name="dummy",
        num_classes=2,
        class_to_idx=fake_classes,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    dm.trainer = SimpleNamespace(world_size=1, state=SimpleNamespace(stage="test"))
    dm.setup(stage="test")

    assert dm.class_to_idx == fake_classes


def test_classification_datamodule_rejects_mismatched_class_to_idx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured class_to_idx that disagrees with what the dataset actually
    contains must fail loudly at setup() time, not silently mislabel results."""
    fake_classes = {"cat": 0, "dog": 1}
    wrong_classes = {"cat": 1, "dog": 0}
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        _fake_load_dataset_with_classes(fake_classes),
    )

    dm = ClassificationImageDataModule(
        dataset_name="dummy",
        num_classes=2,
        class_to_idx=wrong_classes,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    dm.trainer = SimpleNamespace(world_size=1, state=SimpleNamespace(stage="test"))

    with pytest.raises(AssertionError, match="does not match"):
        dm.setup(stage="test")


def test_classification_datamodule_stage_none_with_trainer_builds_test_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the fixed setup() bug: stage=None with a trainer attached
    must resolve the stage from trainer.state and still build the test dataset,
    instead of silently falling through with neither branch firing."""
    fake_classes = {"cat": 0, "dog": 1}
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        _fake_load_dataset_with_classes(fake_classes),
    )

    dm = ClassificationImageDataModule(
        dataset_name="dummy",
        num_classes=2,
        class_to_idx=fake_classes,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    dm.trainer = SimpleNamespace(world_size=1, state=SimpleNamespace(stage="test"))
    dm.setup(stage=None)

    assert dm.data_test is not None
    assert dm.class_to_idx == fake_classes


def _fake_load_dataset_for_artifact() -> dict:
    return {
        "train": [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 0, "image_id": "train-0"}],
        "validation": [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 1, "image_id": "val-0"}],
        "test": [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 2, "image_id": "test-0"}],
    }


def test_artifact_datamodule_fit_stage_never_simulates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Train/val datasets must stay plain HFDataset even when simulate_artifacts_for_test
    is enabled, and the (potentially expensive/network-dependent) artifact pipeline must
    not be built at all for a fit-stage setup()."""
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        lambda _name: _fake_load_dataset_for_artifact(),
    )

    def _fail_if_called(**_kwargs):
        raise AssertionError("build_artifact_pipeline should not be called during fit-stage setup")

    monkeypatch.setattr("src.data.artifact_image_datamodule.build_artifact_pipeline", _fail_if_called)

    dm = ArtifactImageDataModule(
        dataset_name="dummy",
        num_classes=2,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        simulate_artifacts_for_test=True,
    )
    dm.setup(stage="fit")

    assert isinstance(dm.data_train, HFDataset)
    assert isinstance(dm.data_val, HFDataset)
    assert dm.artifact_pipeline is None


def test_artifact_datamodule_test_stage_builds_paired_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test-stage setup() with simulate_artifacts_for_test=True should build
    ArtifactHFDataset, whose batches carry paired real/simulated images."""
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        lambda _name: _fake_load_dataset_for_artifact(),
    )

    def _fake_simulator(image):
        return {"image": image, "metadata": {}}

    monkeypatch.setattr(
        "src.data.artifact_image_datamodule.build_artifact_pipeline",
        lambda **_kwargs: _fake_simulator,
    )

    dm = ArtifactImageDataModule(
        dataset_name="dummy",
        num_classes=2,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        simulate_artifacts_for_test=True,
    )
    dm.setup(stage="test")

    assert isinstance(dm.data_test, ArtifactHFDataset)
    batch = next(iter(dm.test_dataloader()))
    assert "real_image" in batch and "artifact_simulated_image" in batch
    assert batch["real_image"].shape[0] == 1


def test_artifact_datamodule_rejects_mismatched_class_to_idx(monkeypatch: pytest.MonkeyPatch) -> None:
    """ArtifactImageDataModule shares the same configured-vs-actual class_to_idx check
    as ClassificationImageDataModule now that it accepts the same parameter."""
    fake_classes = {"cat": 0, "dog": 1}
    wrong_classes = {"cat": 1, "dog": 0}
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        _fake_load_dataset_with_classes(fake_classes),
    )

    dm = ArtifactImageDataModule(
        dataset_name="dummy",
        num_classes=2,
        class_to_idx=wrong_classes,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        simulate_artifacts_for_test=False,
    )
    dm.trainer = SimpleNamespace(world_size=1, state=SimpleNamespace(stage="test"))

    with pytest.raises(AssertionError, match="does not match"):
        dm.setup(stage="test")


def test_read_meta_rejects_pre_v2_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inference pipeline should reject an old-format checkpoint (no `sngp_core`
    metadata block) early and clearly, rather than silently unpickling whatever
    `hyper_parameters["net"]` happens to be. See src/checkpointing/legacy.py for the
    supported migration path."""

    checkpoint = {"hyper_parameters": {"net": SimpleNamespace()}}
    monkeypatch.setattr("src.checkpointing.io.torch.load", lambda *args, **kwargs: checkpoint)

    with pytest.raises(ValueError, match="sngp_core"):
        read_meta("/tmp/fake.ckpt")
