import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt

# For reproducible results
torch.manual_seed(0)
torch.cuda.manual_seed(0)

import mininet
import datasets

device = 'cuda:0'
batch_size = 32
epochs = 120

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
        break

    zeros = freq == 0
    if zeros.sum() != 0:
        print("There are some classes not present in the training examples")

    result = freq.median() / freq
    result[zeros] = 0 # avoid inf values
    return result ** soft

def train_step(model, train_dataloader, optimizer, criterion, scheduler, device):
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

def main():

    train_data = datasets.Cityscapes('data/cityscapes', split='train', mode='fine', use_train_classes=True, target_type='semantic', transforms=transform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = mininet.MiniNetv2(3, train_data.num_classes, interpolate=True)
    model = model.to(device=device)

    print('Calculating class weights...')
    class_weights = median_frequency_exp(train_dataloader, train_data.num_classes, 0.12)
    print(class_weights)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=train_data.ignore_index).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, epochs, 0.9)

    print('Training...')
    losses = []
    for i in range(epochs):
        train_losses = train_step(model, train_dataloader, optimizer, criterion, scheduler, device)
        losses.extend(train_losses)

        print(f'epoch {i:4d}, {losses[-1]:.4f}')

        # Save model
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'loss': losses
        }, f'model_{i:03d}.tar')

    matplotlib.use('agg')
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

if __name__ == "__main__":
    main()
