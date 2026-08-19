import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset


def _apply_transform(image, transform, mask=None):
    """Apply a torchvision or albumentations transform to a single image.

    Albumentations expects numpy arrays and a separate `image=`/`mask=` call
    signature; torchvision transforms are plain callables on the image alone.
    """
    if transform is None:
        return image

    if isinstance(transform, A.Compose):
        image_np = np.array(image)
        transformed = transform(image=image_np, mask=mask)
        if mask is not None:
            raise NotImplementedError("Mask augmentation is not yet returned")
        return transformed["image"]

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
