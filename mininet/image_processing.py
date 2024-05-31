import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms


def get_train_transforms(size=(256, 512)):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomResizedCrop(size, scale=(0.5, 2), ratio=(2, 2)),
        transforms.ToImage(),
        transforms.ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToPureTensor()
    ])


def get_val_transforms(size=(256, 512)):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToImage(),
        transforms.ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64}, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToPureTensor()
    ])


def denormalize(input):
    input = input * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    input = torch.clip(input, 0, 1)
    return input
