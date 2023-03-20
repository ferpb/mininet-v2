import numpy as np
import random

import torch
from torch.utils.data import DataLoader

# for reproducible results
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg')

import mininet
import loaders


def main():
    # Training parameters
    data_path = 'data/camvid'

    batch_size = 16
    epochs = 200

    results_path = 'output'

    device = 'cuda:0'

    # Load data

    train_data = loaders.CamvidLoader(data_path, 'train', False)
    test_data = loaders.CamvidLoader(data_path, 'test', False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Define and train model

    model = mininet.MiniNetv2(3, train_data.num_classes)
    model = model.cuda(device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    losses = []
    for i in range(epochs):
        running_loss = 0

        for x, y in train_dataloader:
            x = x.cuda(device=device)
            y = y.cuda(device=device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            y = model(x)

        print(f'\r{i:4d}, {running_loss:.4f}', end='')

        loss = running_loss / len(train_dataloader)
        losses.append(loss)

    # Save model

    torch.save({
        'epoch': i,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }, 'model.tar')

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

if __name__ == "__main__":
    main()
