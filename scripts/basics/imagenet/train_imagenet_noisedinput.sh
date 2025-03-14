

#!/bin/bash
# sh scripts/basics/cifar100/train_cifar100.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \

CUDA_VISIBLE_DEVICES=2 python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/networks/resnet50.yml\
    configs/pipelines/train/train_noisedinput.yml \
    --seed 0