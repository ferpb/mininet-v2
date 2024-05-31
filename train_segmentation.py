import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import argparse

import torchmetrics

import matplotlib
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed(0)

import mininet
import datasets
import utils

from torch.utils.tensorboard import SummaryWriter


def train_epoch(model, dataloader, optimizer, criterion, lr_scheduler, device, epoch, writer=None):
    model.train()

    metric_loss = torchmetrics.MeanMetric().to(device)
    metric_iou = torchmetrics.JaccardIndex(
        num_classes=dataloader.dataset.num_classes,
        ignore_index=dataloader.dataset.ignore_index,
        task='multiclass',
        average='none',
    ).to(device)

    for i, (input, target) in enumerate(dataloader):
        input = input.to(device, non_blocking=True)
        target = target.squeeze(1).to(device, non_blocking=True)
        logits = model(input)
        loss = criterion(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_loss.update(loss)
        metric_iou.update(logits, target)

        writer.add_scalar('train/batch_loss', loss, epoch * len(dataloader) + i)

    writer.add_scalar('train/lr_scheduler', lr_scheduler.get_last_lr()[0], epoch)
    lr_scheduler.step()

    print('learning_rate', lr_scheduler.get_last_lr()[0])

    return metric_loss.compute(), metric_iou.compute()


def validate(model, dataloader, criterion, device, writer=None):
    model.eval()

    metric_loss = torchmetrics.MeanMetric().to(device)
    metric_iou = torchmetrics.JaccardIndex(
        num_classes=dataloader.dataset.num_classes,
        ignore_index=dataloader.dataset.ignore_index,
        task='multiclass',
        average='none'
    ).to(device)

    with torch.inference_mode():
        for input, target in dataloader:
            input = input.to(device, non_blocking=True)
            target = target.squeeze(1).to(device, non_blocking=True)
            logits = model(input)
            loss = criterion(logits, target)
            metric_loss.update(loss)
            metric_iou.update(logits, target)

    return metric_loss.compute(), metric_iou.compute()


def main(args):
    train_transforms = mininet.get_train_transforms((256, 512))
    train_data = datasets.Cityscapes(args.dataset_path, split='train', mode='fine', use_train_classes=True, target_type='semantic', transforms=train_transforms)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)

    val_transforms = mininet.get_val_transforms((256, 512))
    val_data = datasets.Cityscapes(args.dataset_path, split='val', mode='fine', use_train_classes=True, target_type='semantic', transforms=val_transforms)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = mininet.MiniNetV2Segmentation(3, train_data.num_classes, interpolate=True)
    model = model.to(device=args.device)

    print('Calculating class weights...')
    class_weights = utils.median_frequency_exp(train_data, train_data.num_classes, 0.12)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=train_data.ignore_index).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, args.max_epoch, 0.9)

    if args.resume:
        # Resume training
        print('Loading checkpoint...')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.finetune:
        # Start from a pretrained model
        print('Loading checkpoint...')
        checkpoint = torch.load(args.finetune)
        model.load_state_dict(checkpoint['model'])

    if args.test_only:
        val_loss, val_iou = validate(model, val_dataloader, criterion, args.device)
        val_miou = val_iou.mean()
        print(f'epoch: {args.start_epoch}, val_loss: {val_loss:.4f}, val_miou: {val_miou:.4f}')
        return

    print('Training...')

    root = Path(f'{args.results_path}/segmentation_cityscapes_{datetime.now().strftime("%s")}')
    root.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(root)

    best_val_miou = 0

    for epoch in range(args.start_epoch, args.max_epoch):
        train_loss, train_iou =  train_epoch(model, train_dataloader, optimizer, criterion, lr_scheduler, args.device, epoch, writer)
        val_loss, val_iou = validate(model, val_dataloader, criterion, args.device, writer)

        train_miou = train_iou.mean()
        val_miou = val_iou.mean()

        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/miou', train_miou, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/miou', val_miou, epoch)

        print(f'epoch: {epoch:03d}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')

        utils.save_checkpoint(root / f'checkpoint_{epoch:03d}.tar', model, optimizer, lr_scheduler, epoch, args)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            utils.save_checkpoint(root / 'best_checkpoint.tar', model, optimizer, lr_scheduler, epoch, args)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='data/cityscapes')
    parser.add_argument('--results-path', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--batch-size', type=int, default=12)

    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--max-epoch', type=int, default=250)

    parser.add_argument('--resume', type=str)
    parser.add_argument('--finetune', type=str)

    parser.add_argument('--test-only', action='store_true')

    args = parser.parse_args()
    main(args)
