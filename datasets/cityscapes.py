import torch
import torchvision
import numpy as np
from PIL import Image

from pathlib import Path
from typing import Tuple, Any, Union, Optional, Callable

from torchvision import tv_tensors


class Cityscapes(torchvision.datasets.Cityscapes):
    """
    Improved version to load train labels and return the images into tv_tensors
    """
    def __init__(self, *args, use_train_classes=False, **kwargs):
        self.use_train_classes = use_train_classes

        super().__init__(*args, **kwargs)

        if self.use_train_classes:
            self.num_classes = 19
            self.ignore_index = 255
        else:
            self.num_classes = len(self.classes) - 2 # discard -1 and 0 classes
            self.ignore_index = 0

        self.train_labels_map = {c.train_id: c.id for c in self.classes}
        self.train_labels_map[255] = 0

    def from_train_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.apply_(self.train_labels_map.get)

    def color_labels(self, labels: torch.Tensor) -> np.array:
        result = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        for c in self.classes:
            if c.train_id == self.ignore_index:
                continue
            result[labels == c.train_id] = c.color
        return result

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'semantic':
            if self.use_train_classes:
                return f'{mode}_labelTrainIds.png'
            else:
                return f'{mode}_labelIds.png'

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        image = tv_tensors.Image(Image.open(self.images[index]).convert("RGB"))

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = tv_tensors.Mask(Image.open(self.targets[index][i]))

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
