#!/bin/sh

mkdir -p data/cityscapes
cd data/cityscapes
csDownload gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
unzip -n gtFine_trainvaltest.zip
unzip -n leftImg8bit_trainvaltest.zip
cd -
CITYSCAPES_DATASET=data/cityscapes csCreateTrainIdLabelImgs
