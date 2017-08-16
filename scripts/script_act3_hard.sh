#!/bin/bash

"""
CUDA_VISIBLE_DEVICES=1 python train.py --seed 0 --house 0 --update-freq 20 \
    --max-episode-len 100 --linear-reward --replay-buffer-size 2000000 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/act3/house0/linear_reward/hard/bc_eplen100 \
    --log-dir ./log/act3/house0/linear_reward/hard/bc_eplen100 \
    --batch-size 256 --hardness 0.95 \
    --critic-weight-decay 0.00001 --critic-penalty 0.0001 --batch-norm \
    --noise-scheduler high --action-dim 3 --no-debug
"""
    #--entropy-penalty 0.001 \

CUDA_VISIBLE_DEVICES=1 python train.py --seed 0 --house 0 --update-freq 20 \
    --max-episode-len 100 --linear-reward --replay-buffer-size 2000000 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/act3/house0/linear_reward/hard/exp_low_eplen100 \
    --log-dir ./log/act3/house0/linear_reward/hard/exp_low_eplen100 \
    --batch-size 256 --hardness 0.95 \
    --critic-weight-decay 0.00001 --critic-penalty 0.0001 --batch-norm \
    --noise-scheduler low --action-dim 3 --no-debug
