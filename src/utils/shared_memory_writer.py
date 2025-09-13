#!/usr/bin/env python3
"""shared_memory_writer.py in src/sngp_core/models."""

import os
from multiprocessing.shared_memory import SharedMemory
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
from numpy.typing import NDArray


class SharedMemoryWriter(pl.callbacks.BasePredictionWriter):
    """Writes multi-GPU predictions to shared memory."""

    def __init__(self, num_samples: int, num_classes: int, num_features: int) -> None:
        """Create a new SharedMemoryWriter callback.

        Args:
            num_samples (int): number of samples in the dataset.
            num_classes (int): number of classes in the dataset.
            num_features (int): number of features in the dataset.
        """
        super().__init__(write_interval="batch")
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_features = num_features
        self.local_rank = 0

        (
            self.feat_shm,
            self.label_shm,
            self.img_id_shm,
            self.inst_id_shm,
            self.img_md5_shm,
            self.img_file_shm,
        ) = self._get_shm()
        (
            self.feats,
            self.labels,
            self.img_ids,
            self.inst_ids,
            self.image_md5,
            self.img_files,
        ) = self._get_arrays()

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Any,
        batch_indices: Optional[Sequence[Any]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Write predictions from each process to shared memory."""
        self.local_rank = trainer.local_rank  # for SHM cleanup later
        self.feats[batch_indices] = predictions[0]
        self.labels[batch_indices] = predictions[1]
        self.img_ids[batch_indices] = predictions[2]
        self.inst_ids[batch_indices] = predictions[3]
        self.inst_ids[batch_indices] = predictions[4]
        self.img_files[batch_indices] = predictions[5]

    def get_predictions(
        self,
    ) -> Tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.str_],
        NDArray[np.str_],
        NDArray[np.str_],  # image_md5
        NDArray[np.str_],
    ]:
        """Return prediction vectors."""
        return (
            self.feats,
            self.labels,
            self.img_ids,
            self.inst_ids,
            self.image_md5,
            self.img_files,
        )

    def close(self) -> None:
        """Release shared memory.

        Only call this from the rank 0 process as multiple calls to close will
        raise an exception.
        """
        if self.local_rank == 0:
            self.feat_shm.close()
            self.feat_shm.unlink()
            self.label_shm.close()
            self.label_shm.unlink()
            self.img_id_shm.close()
            self.img_id_shm.unlink()
            self.inst_id_shm.close()
            self.inst_id_shm.unlink()
            self.img_md5_shm.close()
            self.img_md5_shm.unlink()
            self.img_file_shm.close()
            self.img_file_shm.unlink()

    def _get_names(self) -> Tuple[str, str, str, str, str, str]:
        """Get a unique name for shared memory blocks.

        Distributed Data Parallel creates copies of the parent process. We want all
        instances of the SharedMemoryWriter to write to the same block of shared
        memory. The parent and child processes all share the same process group ID.
        This ID is used to create a common name for all the parallel processes.
        """
        pgid = os.getpgid(0)
        return (
            f"feature-{pgid}",
            f"label-{pgid}",
            f"image_id-{pgid}",
            f"instance_id-{pgid}",
            f"image_md5-{pgid}",
            f"image_file-{pgid}",
        )

    def _get_shm(
        self,
    ) -> Tuple[
        SharedMemory,
        SharedMemory,
        SharedMemory,
        SharedMemory,
        SharedMemory,
        SharedMemory,
    ]:
        """Get the shared memory blocks for the SharedMemoryWriter.

        The first process to enter this section will attempt to allocate the block
        of memory. If a process fails to create a block because it already exists,
        it will simply return a handle to that block.
        """
        (
            feature_name,
            label_name,
            img_id_name,
            inst_id_name,
            img_md5_name,
            img_file_name,
        ) = self._get_names()

        # features
        try:
            feature_shm = SharedMemory(
                create=True,
                size=4 * self.num_samples * self.num_features,
                name=feature_name,
            )
        except FileExistsError:
            feature_shm = SharedMemory(feature_name)

        # label
        try:
            label_shm = SharedMemory(
                create=True, size=8 * self.num_samples, name=label_name
            )
        except FileExistsError:
            label_shm = SharedMemory(label_name)

        # image_id
        try:
            img_id_shm = SharedMemory(
                create=True, size=32 * self.num_samples, name=img_id_name
            )
        except FileExistsError:
            img_id_shm = SharedMemory(img_id_name)

        # instance_id
        try:
            inst_id_shm = SharedMemory(
                create=True, size=32 * self.num_samples, name=inst_id_name
            )
        except FileExistsError:
            inst_id_shm = SharedMemory(inst_id_name)

        # image_md5
        try:
            img_md5_shm = SharedMemory(
                create=True, size=32 * self.num_samples, name=img_md5_name
            )
        except FileExistsError:
            img_md5_shm = SharedMemory(img_md5_name)

        # image_file
        try:
            img_file_shm = SharedMemory(
                create=True, size=32 * self.num_samples, name=img_file_name
            )
        except FileExistsError:
            img_file_shm = SharedMemory(img_file_name)

        return (
            feature_shm,
            label_shm,
            img_id_shm,
            inst_id_shm,
            img_md5_shm,
            img_file_shm,
        )

    def _get_arrays(
        self,
    ) -> Tuple[
        NDArray[np.float32],
        NDArray[np.int64],
        NDArray[np.str_],
        NDArray[np.str_],
        NDArray[np.str_],
        NDArray[np.str_],
    ]:
        """Create NumPy arrays backed by a shared memory block."""
        feats: NDArray[np.float32] = np.ndarray(
            (self.num_samples, self.num_features),
            dtype=np.float32,
            buffer=self.feat_shm.buf,
        )
        labels: NDArray[np.int32] = np.ndarray(
            (self.num_samples, 1), dtype=np.int32, buffer=self.label_shm.buf
        )
        img_ids: NDArray[np.str_] = np.ndarray(
            (self.num_samples, 1), dtype=np.str_, buffer=self.img_id_shm.buf
        ).astype("<U40")
        inst_ids: NDArray[np.str_] = np.ndarray(
            (self.num_samples, 1), dtype=np.str_, buffer=self.inst_id_shm.buf
        ).astype("<U40")
        img_md5: NDArray[np.str_] = np.ndarray(
            (self.num_samples, 1), dtype=np.str_, buffer=self.img_md5_shm.buf
        ).astype("<U40")
        img_files: NDArray[np.str_] = np.ndarray(
            (self.num_samples, 1), dtype=np.str_, buffer=self.img_file_shm.buf
        ).astype("<U256")

        return feats, labels, img_ids, inst_ids, img_md5, img_files
