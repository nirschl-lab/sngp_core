from time import sleep
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

from dotenv import load_dotenv, find_dotenv
env_path = find_dotenv()
load_dotenv(env_path)

# Custom Dataset wrapper
class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']  # PIL Image
        if self.transform:
            image = self.transform(image)
        return image

def build_transform(image_size: int = 299) -> transforms.Compose:
    """
    FID is computed with Inception-v3 features.
    Inception expects 299x299 and pixel values in [0,1].
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),                     # [0,1]
    ])



@torch.no_grad()
def accumulate_fid_stream(fid: FrechetInceptionDistance, loader: DataLoader, is_real: bool, device: torch.device):
    for batch in loader:
        # batch could be (images, labels) if ImageFolder; handle tuple
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        imgs = imgs.to(device, non_blocking=True)
        # TorchMetrics expects uint8 images in [0,255] OR float in [0,1]? (Both are supported.)
        # We'll convert to uint8 [0,255] to be explicit.
        imgs_uint8 = (imgs.clamp(0, 1) * 255).to(torch.uint8)
        fid.update(imgs_uint8, real=is_real)

def calcluate_fid(dataset1: Dataset, dataset2: Dataset, batch_size: int = 64, num_workers: int = 4, device: str = 'cuda') -> float:
    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    fid = FrechetInceptionDistance(feature=2048).to(device)

    accumulate_fid_stream(fid, loader1, is_real=True, device=device)
    accumulate_fid_stream(fid, loader2, is_real=False, device=device)

    fid_value = fid.compute().item()
    return fid_value

if __name__ == '__main__':
    datasets = ['nirschl-lab/jung_et_al_2022',
                    'nirschl-lab/kather_et_al_2016',
                    'nirschl-lab/nirschl_et_al_2018',
                    'nirschl-lab/wong_et_al_2022',
                    'nirschl-lab/tang_et_al_2019',
                    'nirschl-lab/acevedo_et_al_2020']

    fold = 'test' #train
    
    ix1 = 4
    ix2 = 5

    d1 = load_dataset(datasets[ix1])
    d2 = load_dataset(datasets[ix2])

    DL1 = HFDataset(d1[fold], transform=build_transform(299))
    DL2 = HFDataset(d2[fold], transform=build_transform(299))
    fid_value = calcluate_fid(DL1, DL2, batch_size=2048, num_workers=16, device='cuda')
    print(f"FID between {datasets[ix1]} and {datasets[ix2]}: {fid_value}")
    