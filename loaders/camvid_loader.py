import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

import imageio
from pathlib import Path

## Data augmentation

class CamvidLoader(Dataset):
    def __init__(self, root, split, transform=None):
        self.num_classes = 12
        self.split = split
        self.transform = transform

        images_paths = sorted(Path(root, "images", split).iterdir())
        labels_paths = sorted(Path(root, "labels", split).iterdir())

        self.x = [imageio.imread(file) for file in images_paths]
        self.y = [imageio.imread(file) for file in labels_paths]

        print(f"Loaded {len(self.x)} images ({split})")

        self.x = [x.astype(np.float32) / 255.0 for x in self.x]
        self.y = [y.astype(np.uint8) for y in self.y]

        self.resize = transforms.Resize((360, 240), antialias=True)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        x = torch.tensor(x).permute(2, 0, 1)
        y = torch.tensor(y).unsqueeze(0)

        x = self.resize(x)
        y = self.resize(y)

        y = torch.nn.functional.one_hot(y.long().squeeze(), num_classes=self.num_classes)
        y = y.permute(2, 0, 1).to(torch.float32)

        return x, y
