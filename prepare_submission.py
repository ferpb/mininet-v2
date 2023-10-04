import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt

import mininet
import torchmetrics

from train import normalize
import mininet
import datasets
import imageio.v3 as imageio
from pathlib import Path


transform = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToImageTensor(),
])

device = 'cuda:0'

test_data = datasets.Cityscapes('data/cityscapes', split='val', mode='fine', target_type='semantic', transforms=transform)

model = mininet.MiniNetv2(3, 19, interpolate=True)

model.load_state_dict(torch.load('models/mininet_cityscapes.tar')['model_state_dict'])

model = model.to(device)
model.eval()

root = Path('results')
root.mkdir(exist_ok=True)

for i in range(len(test_data)):
    path = Path(test_data.images[i])

    print(path)
    input, _ = test_data[i]
    input = normalize(input).unsqueeze(dim=0).to(device='cuda:1')
    pred = model(input)
    pred = pred.argmax(dim=1).cpu()
    pred = test_data.from_train_labels(pred)
    pred = transforms.functional.resize(pred, (1024, 2048))

    print(pred.min(), pred.max())
    pred = pred.to(dtype=torch.uint8)

    imageio.imwrite(root / f'{path.stem}.png', pred)
