import h5py
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

class H5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.transform = transform
        self.dataset = None
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f)
            self.keys = [
                key for key in f.keys()
                if f[key].shape[0] >= 512 and f[key].shape[1] >= 512
            ]

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        key = self.keys[index]
        image = self.dataset[key][:]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.keys)
