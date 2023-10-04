import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import argparse

import matplotlib
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed(0)

import mininet
import datasets

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.05),
    transforms.RandomResizedCrop((256, 512), scale=(0.5, 2), ratio=(2, 2), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToImageTensor(),
])

def normalize(batch):
    batch = batch.to(dtype=torch.float32) / 255 - 0.5
    return batch

def median_frequency_exp(dataloader, num_classes, soft):
    freq = torch.zeros(num_classes, dtype=torch.long)

    for _, y in dataloader:
        classes, counts = y.unique(return_counts=True)
        # ignore extra classes
        ignore = torch.bitwise_or(classes < 0, classes >= num_classes)
        # add counts
        freq = freq.scatter_add(0, classes[~ignore].long(), counts[~ignore])

    zeros = freq == 0
    if zeros.sum() != 0:
        print("There are some classes not present in the training examples")

    result = freq.median() / freq
    result[zeros] = 0 # avoid inf values
    return result ** soft

def train_epoch(model, train_dataloader, optimizer, criterion, scheduler, device):
    model.train()

    losses = []
    for x, y in train_dataloader:
        x = normalize(x).to(device=device)
        y = y.squeeze().long().to(device=device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    scheduler.step()

    return losses

def validate(model, dataloader, criterion, device):
    model.eval()

    with torch.no_grad():
        losses = []
        for x, y in dataloader:
            x = normalize(x).to(device=device)
            y = y.squeeze().long().to(device=device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            losses.append(loss.item())

    return sum(losses) / len(dataloader)

def save_checkpoint(path, epoch, model, train_losses, val_losses):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }, path)

def main(args):
    train_data = datasets.Cityscapes(args.dataset_path, split='train', mode='fine', use_train_classes=True, target_type='semantic', transforms=transform)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = datasets.Cityscapes(args.dataset_path, split='val', mode='fine', use_train_classes=True, target_type='semantic', transforms=transform)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    model = mininet.MiniNetv2(3, train_data.num_classes, interpolate=True)
    model = model.to(device=args.device)

    print('Calculating class weights...')
    class_weights = median_frequency_exp(train_dataloader, train_data.num_classes, 0.12)
    print(class_weights)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=train_data.ignore_index).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, args.epochs, 0.9)

    root = Path(f'{args.results_path}/{datetime.now().strftime("%s")}')
    root.mkdir(parents=True, exist_ok=True)

    print('Training...')
    train_losses = []
    val_losses = []
    for i in range(args.epochs):
        train_epoch(model, train_dataloader, optimizer, criterion, scheduler, args.device)

        train_loss = validate(model, train_dataloader, criterion, args.device)
        val_loss = validate(model, val_dataloader, criterion, args.device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'epoch {i:4d}, {train_loss:.4f}, {val_loss:.4f}')

        if i % 10 == 0:
            save_checkpoint(root / f'checkpoint_{i:03d}.tar', i, model, train_losses, val_losses)

    save_checkpoint(root / f'final_checkpoint_{i:03d}.tar', i, model, train_losses, val_losses)

    matplotlib.use('agg')
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(root / 'loss.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='data/cityscapes')
    parser.add_argument('--results-path', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    main(args)
