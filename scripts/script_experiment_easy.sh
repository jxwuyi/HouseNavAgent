#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python train.py --seed 0 --house 2 --update-freq 25 \
    --max-episode-len 100 --linear-reward --replay-buffer-size 1000000 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/house_2/linear_reward/easy/bc_eplen100 \
    --log-dir ./log/house_2/linear_reward/easy/bc_eplen100 \
    --batch-size 256 --hardness 0.3 \
    --critic-weight-decay 0.000001 --critic-penalty 0.0001 --batch-norm --no-debug
