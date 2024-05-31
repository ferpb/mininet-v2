import argparse
import torch
from torch.utils.data import Dataset, DataLoader


def median_frequency_exp(dataset: Dataset, num_classes: int, soft: float):
    # Process the dataset in parallel
    loader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False)

    # Initialize counts
    classes_freqs = torch.zeros(num_classes, dtype=torch.int64)

    for _, target in loader:
        classes, counts = torch.unique(target, return_counts=True)
        ignore = torch.bitwise_or(classes < 0, classes >= num_classes)
        classes_freqs.index_add_(0, classes[~ignore], counts[~ignore])

    zeros = classes_freqs == 0
    if zeros.sum() != 0:
        print("There are some classes not present in the training samples")

    result = classes_freqs.median() / classes_freqs
    result[zeros] = 0  # avoid inf values
    return result ** soft


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': epoch,
        'args': args
    }
    torch.save(checkpoint, path)
