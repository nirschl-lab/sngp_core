from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torch
import numpy as np
import albumentations as A
import pdb
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, fold=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.fold = fold

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        try:
            image = item['image']  # PIL Image
            label = item['label']
            mask = item.get('mask', None)  # Optional mask
            if mask is not None:
                mask = torch.tensor(mask, dtype=torch.float32)

            image_id = item['image_id']
        except TypeError as e:
            pdb.set_trace()
            raise
        if self.transform:
            # Apply transformations
            if isinstance(self.transform, A.Compose):
                # Albumentations expects numpy arrays
                image_np = np.array(image)
                transformed = self.transform(image=image_np, mask=mask)
                if mask is not None:
                    raise NotImplementedError("Mask augmentation is not yet returned")
                    #image, mask = transformed['image'], transformed['mask']
                else:
                    image = transformed['image']
            else:
                image = self.transform(image)

        return image_id, image, label, self.fold