import torch
import torchvision
import numpy as np

class Cityscapes(torchvision.datasets.Cityscapes):
    def __init__(self, *arg, use_train_classes=False, **kwargs):
        self.use_train_classes = use_train_classes

        super().__init__(*arg, **kwargs)

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
