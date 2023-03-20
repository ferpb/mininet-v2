import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

import imageio.v3 as imageio
from pathlib import Path
import matplotlib.pyplot as plt

labels = {
     0: 'unlabeled',
     1: 'ego vehicle',
     2: 'rectification border',
     3: 'out of roi',
     4: 'static',
     5: 'dynamic',
     6: 'ground',
     7: 'road',
     8: 'sidewalk',
     9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
    -1: 'license plate'
}

class CityScapesLoader(Dataset):
    def __init__(self, root, split, coarse_labels=True, transform=None):
        # labelIds: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        # instaceIds: 1000*class_id + instance_id

        self.num_classes = 35
        self.split = split
        self.transform = transform

        labels_type = 'gtCoarse' if coarse_labels else 'gtFine'
        paths = list(Path(root, 'leftImg8bit', split).glob('*/*.png'))[:200]

        images_paths = []
        labels_paths = []
        for path in paths:
            city, id1, id2, _ = path.stem.split('_')
            label_path = Path(root, labels_type, split, city, f'{city}_{id1}_{id2}_{labels_type}_labelIds.png')
            if label_path.exists():
                images_paths.append(path)
                labels_paths.append(label_path)

        # load images and labels
        self.x = [imageio.imread(file) for file in images_paths]
        self.y = [imageio.imread(file) for file in labels_paths]

        print(f"Loaded {len(self.x)} images ({split})")

        self.x = [x.astype(np.float32) / 255.0 for x in self.x]
        self.y = [y.astype(np.uint8) + 1 for y in self.y]

        # self.resize = transforms.Resize((512, 1024), antialias=True)
        self.resize = transforms.Resize((256, 512), antialias=True)

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
