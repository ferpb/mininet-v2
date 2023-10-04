import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms

from train import normalize
import mininet
import datasets

import imageio.v3 as imageio
from pathlib import Path
import argparse


def main(args):

    transform = transforms.Compose([
        transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToImageTensor(),
    ])

    test_data = datasets.Cityscapes(args.dataset_path, split='val', mode='fine', target_type='semantic', transforms=transform)

    model = mininet.MiniNetv2(3, 19, interpolate=True)

    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])

    model = model.to(args.device)
    model.eval()

    root = Path(f'{args.model_path.split(".")[0]}_test')
    root.mkdir(exist_ok=True)

    for i in range(len(test_data)):
        path = Path(test_data.images[i])

        print(path)
        input, _ = test_data[i]
        input = normalize(input).unsqueeze(dim=0).to(device=args.device)
        pred = model(input)
        pred = pred.argmax(dim=1).cpu()
        pred = test_data.from_train_labels(pred)
        pred = transforms.functional.resize(pred, (1024, 2048))
        pred = pred.to(dtype=torch.uint8)

        imageio.imwrite(root / f'{path.stem}.png', pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='data/cityscapes')
    parser.add_argument('--model-path', type=str, default='pretrained/mininet_cityscapes.tar')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    main(args)
