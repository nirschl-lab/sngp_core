"""test_datamodules.py in tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.checkpointing.io import read_meta
from src.data.classification_image_datamodule import ClassificationImageDataModule
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
        "src.data.classification_image_datamodule.datasets.load_dataset",
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


def test_read_meta_rejects_pre_v2_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The inference pipeline should reject an old-format checkpoint (no `sngp_core`
    metadata block) early and clearly, rather than silently unpickling whatever
    `hyper_parameters["net"]` happens to be. See src/checkpointing/legacy.py for the
    supported migration path."""

    checkpoint = {"hyper_parameters": {"net": SimpleNamespace()}}
    monkeypatch.setattr("src.checkpointing.io.torch.load", lambda *args, **kwargs: checkpoint)

    with pytest.raises(ValueError, match="sngp_core"):
        read_meta("/tmp/fake.ckpt")
