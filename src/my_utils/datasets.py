# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import random
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

        self.samples = sorted([
            f for f in splitdir.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ])

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {splitdir}")

        self.transform = transform
        print(f"[{split.upper()}] Loaded {len(self.samples)} images from {splitdir}")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

        path = self.samples[index]
        
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert("RGB")
                img.load()
            if self.transform:
                return self.transform(img)
            return img
        except Exception as e:
            print(f"Warning: Corrupt image at {path}. Error: {e}. Resampling...")
            new_index = random.randint(0, len(self) - 1)
            return self.__getitem__(new_index)

    def __len__(self):
        return len(self.samples)
