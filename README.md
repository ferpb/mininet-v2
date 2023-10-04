# PyTorch implementation of MiniNet-v2

This repository contains a PyTorch implementation of MiniNet-v2 for semantic segmentation.

## Setup

First, you can install the dependiencies using a virtual environment:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Dataset

We provide a script for downloading the Cityscapes dataset using the `cityscapesscripts` package. You will need an account in the [Cityscapes website](https://www.cityscapes-dataset.com/) to access the data.

```
bash download_datasets.sh
```

## Training

Once the dataset is downloaded, you can train a model with default settings running the training script:

```
python train.py
```

This script by default will save checkpoints during the training in `results/<date>/checkpoint_<epoch>.tar`.

## Inference

The `inference.ipynb` notebook contains an example of how to use the model for segmentation tasks. We also include a script to prepare a submission to the [Cityscapes evaluation server](https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-taskttps://www.cityscapes-dataset.com) using the test images:

```
python prepare_submission.py --model-path results/<date>/checkpoint_<epoch>.tar
```

## Citation

If you find this code useful in your research, please cite the original paper:

```bibtex
@article{alonso2020mininet,
  title = {{MiniNet}: An Efficient Semantic Segmentation {ConvNet} for Real-Time Robotic Applications},
  author = {Alonso, I{\~n}igo and Riazuelo, Luis and Murillo, Ana C.},
  journal = {IEEE Transactions on Robotics (T-RO)},
  year = {2020},
  volume = {36},
  number = {4},
  pages = {1340--1347},
  doi = {10.1109/TRO.2020.2974099},
}
```
