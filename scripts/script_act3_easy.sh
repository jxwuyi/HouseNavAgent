#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py --seed 0 --house 0 --update-freq 25 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 1000000 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/act3/house0/linear_reward/easy/bc_eplen50 \
    --log-dir ./log/act3/house0/linear_reward/easy/bc_eplen50 \
    --batch-size 256 --hardness 0.3 \
    --critic-weight-decay 0.000001 --critic-penalty 0.0001 --batch-norm \
    --action-dim 3 --no-debug

    #--entropy-penalty 0.001 \
