import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset


def _apply_transform(image, transform, mask=None, return_mask: bool = False):
    """Apply a torchvision or albumentations transform to a single image.

    Albumentations expects numpy arrays and a separate `image=`/`mask=` call
    signature; torchvision transforms are plain callables on the image alone.

    With `return_mask=True` the co-transformed mask is returned alongside the image, as
    `(image, mask)`. That is what keeps a segmentation-style mask aligned with its image
    through spatial transforms -- `ArtifactImageDataModule` uses it so the simulator's
    artifact mask goes through the same resize and crop as the image it describes.
    Albumentations only; torchvision's v1 `Compose` cannot co-transform a mask.
    """
    if transform is None:
        return (image, mask) if return_mask else image

    if isinstance(transform, A.Compose):
        image_np = np.array(image)
        transformed = transform(image=image_np, mask=mask)
        if return_mask:
            return transformed["image"], transformed["mask"]
        if mask is not None:
            raise NotImplementedError("Mask augmentation is not yet returned")
        return transformed["image"]

    if return_mask:
        raise NotImplementedError(
            f"return_mask=True needs an albumentations Compose to co-transform the mask; "
            f"got {type(transform).__name__}."
        )
    return transform(image)


class HFDataset(Dataset):
    """Wraps a HuggingFace `datasets` split for use as a `torch.utils.data.Dataset`."""

    def __init__(self, hf_dataset, transform=None, fold=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.fold = fold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]  # PIL Image
        label = item["label"]
        mask = item.get("mask", None)  # Optional mask
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float32)
        image_id = item["image_id"]

        image = _apply_transform(image, self.transform, mask=mask)

        return image_id, image, label, self.fold
