"""test_datamodules.py in tests."""

import collections

import json
from pathlib import Path
from types import SimpleNamespace

import datasets
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


def _fake_load_dataset_with_institutions():
    """Real `datasets.Dataset` splits (not plain lists) carrying an `institution`
    column, needed because `.filter()`/`.column_names` are Dataset-only APIs."""

    def make_split(rows):
        return datasets.Dataset.from_list(rows)

    def fake_load_dataset(_dataset_name: str):
        return {
            "train": make_split([
                {
                    "image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "label": 0,
                    "image_id": "train-0",
                    "institution": "ucdavis",
                },
                {
                    "image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "label": 1,
                    "image_id": "train-1",
                    "institution": "stanford",
                },
            ]),
            "validation": make_split([
                {
                    "image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "label": 0,
                    "image_id": "val-0",
                    "institution": "ucdavis",
                },
                {
                    "image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "label": 1,
                    "image_id": "val-1",
                    "institution": "stanford",
                },
            ]),
            "test": make_split([
                {
                    "image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "label": 0,
                    "image_id": "test-0",
                    "institution": "ucdavis",
                },
                {
                    "image": np.zeros((16, 16, 3), dtype=np.uint8),
                    "label": 1,
                    "image_id": "test-1",
                    "institution": "stanford",
                },
            ]),
        }

    return fake_load_dataset


def test_classification_datamodule_institution_none_loads_all_institutions(monkeypatch: pytest.MonkeyPatch) -> None:
    """No institution id -- every row across institutions is loaded, matching today's
    (pre-filter) behavior."""
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        _fake_load_dataset_with_institutions(),
    )

    dm = ClassificationImageDataModule(dataset_name="dummy", batch_size=1, num_workers=0, pin_memory=False)
    dm.setup(stage="fit")

    assert len(dm.data_train) == 2
    assert {dm.data_train.dataset[i]["institution"] for i in range(2)} == {"ucdavis", "stanford"}


def test_classification_datamodule_institution_narrows_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured institution id narrows train/val/test to only its own rows."""
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        _fake_load_dataset_with_institutions(),
    )

    dm = ClassificationImageDataModule(
        dataset_name="dummy", batch_size=1, num_workers=0, pin_memory=False, institution="ucdavis"
    )
    dm.setup(stage="fit")
    assert len(dm.data_train) == 1
    assert dm.data_train.dataset[0]["institution"] == "ucdavis"
    assert len(dm.data_val) == 1
    assert dm.data_val.dataset[0]["institution"] == "ucdavis"

    dm.setup(stage="test")
    assert len(dm.data_test) == 1
    assert dm.data_test.dataset[0]["institution"] == "ucdavis"


def test_classification_datamodule_institution_not_present_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An institution id absent from the data must fail loudly, not silently train on
    an empty split."""
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        _fake_load_dataset_with_institutions(),
    )

    dm = ClassificationImageDataModule(
        dataset_name="dummy", batch_size=1, num_workers=0, pin_memory=False, institution="nowhere"
    )

    with pytest.raises(ValueError, match="matched zero rows"):
        dm.setup(stage="fit")


def test_classification_datamodule_institution_requested_but_column_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requesting an institution filter against a dataset with no institution column
    at all must fail loudly rather than silently ignoring the filter."""

    def fake_load_dataset(_dataset_name: str):
        return {
            "train": datasets.Dataset.from_list(
                [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 0, "image_id": "train-0"}]
            ),
            "validation": datasets.Dataset.from_list(
                [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 0, "image_id": "val-0"}]
            ),
            "test": datasets.Dataset.from_list(
                [{"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 0, "image_id": "test-0"}]
            ),
        }

    monkeypatch.setattr("src.data.base_image_datamodule.datasets.load_dataset", fake_load_dataset)

    dm = ClassificationImageDataModule(
        dataset_name="dummy", batch_size=1, num_workers=0, pin_memory=False, institution="ucdavis"
    )

    with pytest.raises(ValueError, match="no 'institution' column"):
        dm.setup(stage="fit")


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
        artifact_bank_dir="/nonexistent-asset-bank",
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


_FAKE_LABEL_NAMES = ("debris", "pigment_ink", "global:illumination_gradient")


def _install_fake_simulator(monkeypatch: pytest.MonkeyPatch, seen_seeds: list | None = None):
    """Stand in for the histo_artifacts pipeline, matching its output contract."""

    def _fake_simulator(image, seed=None, count=None):
        if seen_seeds is not None:
            seen_seeds.append(seed)
        height, width = np.asarray(image).shape[:2]
        return {
            "image": image,
            "artifact_mask": np.zeros((height, width), dtype=np.uint8),
            "instance_mask": np.zeros((height, width), dtype=np.uint16),
            "artifact_labels": np.zeros(len(_FAKE_LABEL_NAMES), dtype=np.float32),
            "metadata": {"global_degradations": [], "geometric_degradations": []},
        }

    _fake_simulator.label_names = _FAKE_LABEL_NAMES
    monkeypatch.setattr(
        "src.data.artifact_image_datamodule.build_artifact_pipeline",
        lambda **_kwargs: _fake_simulator,
    )
    return _fake_simulator


def test_artifact_datamodule_test_stage_builds_paired_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test-stage setup() with simulate_artifacts_for_test=True should build
    ArtifactHFDataset, whose batches carry paired real/simulated images plus the
    artifact mask and multi-hot labels."""
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        lambda _name: _fake_load_dataset_for_artifact(),
    )
    _install_fake_simulator(monkeypatch)

    dm = ArtifactImageDataModule(
        dataset_name="dummy",
        artifact_bank_dir="/nonexistent-asset-bank",
        num_classes=2,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        simulate_artifacts_for_test=True,
    )
    dm.setup(stage="test")

    assert isinstance(dm.data_test, ArtifactHFDataset)
    assert dm.artifact_label_names == _FAKE_LABEL_NAMES

    batch = next(iter(dm.test_dataloader()))
    assert "real_image" in batch and "artifact_simulated_image" in batch
    assert batch["real_image"].shape[0] == 1
    # The mask has to survive the same spatial transforms as the image it describes,
    # so its spatial dims must track the simulated image's rather than the raw input's.
    assert "artifact_mask" in batch
    assert batch["artifact_mask"].shape[-2:] == batch["artifact_simulated_image"].shape[-2:]
    assert batch["artifact_labels"].shape == (1, len(_FAKE_LABEL_NAMES))
    # No overlay in this fake simulator's mask (all zero), so nothing is "affected".
    assert batch["percent_pixels_affected"][0].item() == pytest.approx(0.0)


def test_artifact_datamodule_percent_pixels_affected_thresholds_the_alpha_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`artifact_mask` is per-pixel blend alpha (0-255), not a 0/1 flag -- a soft-edged
    overlay shades off toward its border. percent_pixels_affected has to threshold at >0
    (fraction of pixels touched at all) rather than average the raw alpha, which would
    report a much smaller (and, summed over full-alpha pixels near 255, easily >100%)
    number instead of a real coverage percentage."""

    def _fake_simulator(image, seed=None, count=None):
        height, width = np.asarray(image).shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        # Half the pixels get a low, non-255 alpha -- still "affected", not "half-affected".
        mask[: height // 2, :] = 5
        return {
            "image": image,
            "artifact_mask": mask,
            "instance_mask": np.zeros((height, width), dtype=np.uint16),
            "artifact_labels": np.zeros(len(_FAKE_LABEL_NAMES), dtype=np.float32),
            "metadata": {"global_degradations": ["hed_stain_shift"], "geometric_degradations": []},
        }

    _fake_simulator.label_names = _FAKE_LABEL_NAMES
    monkeypatch.setattr(
        "src.data.artifact_image_datamodule.build_artifact_pipeline",
        lambda **_kwargs: _fake_simulator,
    )
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset",
        lambda _name: _fake_load_dataset_for_artifact(),
    )

    dm = ArtifactImageDataModule(
        dataset_name="dummy",
        artifact_bank_dir="/nonexistent-asset-bank",
        num_classes=2,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        simulate_artifacts_for_test=True,
    )
    dm.setup(stage="test")

    batch = next(iter(dm.test_dataloader()))
    assert batch["percent_pixels_affected"][0].item() == pytest.approx(50.0, abs=1.0)
    # global_degradations travels through as a JSON-encoded string per sample (not a raw
    # list) because default_collate can't batch variable-length lists across samples.
    assert json.loads(batch["global_degradations"][0]) == ["hed_stain_shift"]
    assert json.loads(batch["geometric_degradations"][0]) == []


def test_artifact_datamodule_seeds_each_sample_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every sample must get its own content-derived seed.

    Without one, the simulator's single RNG stream is replayed identically inside each
    forked DataLoader worker, so samples at the same within-worker offset receive the
    same artifacts. Keying on image_id also keeps a sample's artifacts stable regardless
    of worker count or shard order."""
    dataset = {
        "test": [
            {"image": np.zeros((16, 16, 3), dtype=np.uint8), "label": 0, "image_id": f"test-{i}"}
            for i in range(3)
        ]
    }
    monkeypatch.setattr(
        "src.data.base_image_datamodule.datasets.load_dataset", lambda _name: dataset
    )
    seen_seeds: list = []
    _install_fake_simulator(monkeypatch, seen_seeds)

    dm = ArtifactImageDataModule(
        dataset_name="dummy",
        artifact_bank_dir="/nonexistent-asset-bank",
        num_classes=2,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        simulate_artifacts_for_test=True,
        artifact_seed=7,
    )
    dm.setup(stage="test")
    list(dm.test_dataloader())

    assert len(seen_seeds) == 3
    assert all(seed is not None for seed in seen_seeds)
    assert len(set(seen_seeds)) == 3, "each image_id must map to a distinct seed"

    # Reproducible: the same artifact_seed and image_ids must derive the same seeds.
    from histo_artifacts.seeding import derive_seed

    assert seen_seeds == [derive_seed(7, f"test-{i}") for i in range(3)]


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
        artifact_bank_dir="/nonexistent-asset-bank",
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


# ---------------------------------------------------------------------------
# BenchmarkImageDataModule (CIFAR-100/10, SVHN)
# ---------------------------------------------------------------------------
# These build real in-memory `datasets.Dataset` objects rather than the list-of-dicts
# fakes above, because the whole point of this subclass is the Arrow-level surgery it
# does -- rename_column / remove_columns / add_column / stratified train_test_split.
# A dict fake would exercise none of it. Still no network: nothing is downloaded.


def _fake_cifar_like(num_classes: int = 4, per_class: int = 10, with_coarse: bool = True):
    """A miniature `uoft-cs/cifar100`: `img` / `fine_label` / `coarse_label`, train+test
    splits, no `image_id`, no `classes_to_idx`, no `validation`."""
    from PIL import Image as PILImage

    names = [f"class_{i}" for i in range(num_classes)]
    features = {
        "img": datasets.Image(),
        "fine_label": datasets.ClassLabel(names=names),
    }
    if with_coarse:
        features["coarse_label"] = datasets.ClassLabel(names=["coarse_a", "coarse_b"])

    def build(n_per_class: int):
        payload = {
            "img": [
                PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
                for _ in range(num_classes * n_per_class)
            ],
            "fine_label": [c for c in range(num_classes) for _ in range(n_per_class)],
        }
        if with_coarse:
            payload["coarse_label"] = [
                c % 2 for c in range(num_classes) for _ in range(n_per_class)
            ]
        return datasets.Dataset.from_dict(payload, features=datasets.Features(features))

    return {"train": build(per_class), "test": build(2)}, names


def _benchmark_dm(monkeypatch: pytest.MonkeyPatch, raw, names, **kwargs):
    from src.data.benchmark_image_datamodule import BenchmarkImageDataModule

    monkeypatch.setattr(
        "src.data.benchmark_image_datamodule.datasets.load_dataset",
        lambda *_args, **_kwargs: raw,
    )
    defaults = dict(
        dataset_name="fake/cifar",
        image_column="img",
        label_column="fine_label",
        drop_columns=["coarse_label"],
        val_split_size=8,
        num_classes=len(names),
        class_to_idx={name: i for i, name in enumerate(names)},
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    return BenchmarkImageDataModule(**{**defaults, **kwargs})


def test_benchmark_datamodule_adapts_schema_and_carves_stratified_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three gaps that stop a stock benchmark loading through the project's schema --
    wrong column names, no `image_id`, no `validation` split -- must all close, and the
    carve must come out of `train` (never `test`) with classes held in proportion."""
    raw, names = _fake_cifar_like(num_classes=4, per_class=10)
    dm = _benchmark_dm(monkeypatch, raw, names)
    dm.setup(stage="fit")

    assert len(dm.data_train) == 32 and len(dm.data_val) == 8  # 40 train, 8 carved out
    assert dm.data_val.dataset.column_names == ["image", "label", "image_id"]
    assert collections.Counter(dm.data_val.dataset["label"]) == {0: 2, 1: 2, 2: 2, 3: 2}

    image_ids, images, labels, fold = next(iter(dm.val_dataloader()))
    assert images.shape[0] == 4 and labels.shape[0] == 4
    assert all(i.startswith("validation-") for i in image_ids)
    assert fold[0] == "validation"


def test_benchmark_datamodule_leaves_an_existing_validation_split_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A benchmark that already ships `validation` must not have a second one carved out
    of its train split."""
    raw, names = _fake_cifar_like(num_classes=4, per_class=10)
    raw = {**raw, "validation": raw["test"]}
    dm = _benchmark_dm(monkeypatch, raw, names)
    dm.setup(stage="fit")

    assert len(dm.data_train) == 40  # untouched
    assert len(dm.data_val) == 8


def test_benchmark_datamodule_ignores_splits_the_base_class_never_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SVHN ships a 531k-row `extra` split. Renaming, indexing and caching it would cost
    minutes and gigabytes for data no stage ever touches."""
    raw, names = _fake_cifar_like(num_classes=4, per_class=10)
    raw = {**raw, "extra": raw["train"]}
    dm = _benchmark_dm(monkeypatch, raw, names)

    assert sorted(dm._load_raw_dataset()) == ["test", "train", "validation"]


def test_benchmark_datamodule_validates_class_to_idx_against_the_class_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """These datasets carry no `classes_to_idx` column, so the base class's cross-check
    cannot run. The ClassLabel feature stands in for it -- a mis-specified mapping in the
    YAML must still fail loudly rather than mislabel every prediction."""
    raw, names = _fake_cifar_like(num_classes=4, per_class=10)
    wrong = {name: i for i, name in enumerate(reversed(names))}
    dm = _benchmark_dm(monkeypatch, raw, names, class_to_idx=wrong)
    dm.trainer = SimpleNamespace(world_size=1, state=SimpleNamespace(stage="test"))

    with pytest.raises(AssertionError, match="does not match the dataset's own ClassLabel"):
        dm.setup(stage="test")


def test_benchmark_datamodule_adopts_the_class_label_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, names = _fake_cifar_like(num_classes=4, per_class=10)
    dm = _benchmark_dm(monkeypatch, raw, names, class_to_idx=None)
    dm.trainer = SimpleNamespace(world_size=1, state=SimpleNamespace(stage="test"))
    dm.setup(stage="test")

    assert dm.class_to_idx == {name: i for i, name in enumerate(names)}
    assert dm.trainer.test_idx_to_classes[0] == names[0]


def test_benchmark_datamodule_rejects_an_impossible_validation_carve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, names = _fake_cifar_like(num_classes=4, per_class=10)
    dm = _benchmark_dm(monkeypatch, raw, names, val_split_size=999)

    with pytest.raises(ValueError, match="val_split_size must be in"):
        dm.setup(stage="fit")
