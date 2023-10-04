import torch
import torchvision

class Cityscapes(torchvision.datasets.Cityscapes):
    def __init__(self, *arg, use_train_classes=False, **kwargs):
        self.use_train_classes = use_train_classes

        super().__init__(*arg, **kwargs)

        if self.use_train_classes:
            self.num_classes = 19
            self.ignore_index = 255
        else:
            self.num_classes = len(self.classes) - 2 # discard -1 and 0 classes
            self.ignore_idex = 0

        self.train_labels_map = {c.train_id: c.id for c in self.classes}

    def from_train_labels(self, labels: torch.Tensor):
        return labels.apply_(self.train_labels_map.get)

    def color_labels(self, labels: torch.Tensor):
        result = torch.zeros((labels.shape[0], labels.shape[1], 3), dtype=torch.uint8)
        for c in self.classes:
            if c.train_id == self.ignore_index:
                continue
            result[labels == c.train_id] = torch.tensor(c.color, dtype=torch.uint8)
        return result

    def _get_target_suffix(self, mode: str, target_type: str):
        if target_type == 'semantic':
            if self.use_train_classes:
                return f'{mode}_labelTrainIds.png'
            else:
                return f'{mode}_labelIds.png'
