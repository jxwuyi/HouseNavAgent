#!/bin/bash

"""
CUDA_VISIBLE_DEVICES=4 python train.py --seed 0 --house 0 --update-freq 10 \
    --max-episode-len 100 --linear-reward --replay-buffer-size 1500000 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/large_net/linear_reward/hard/exp_low_eplen100_freq10 \
    --log-dir ./log/large_net/linear_reward/hard/exp_low_eplen100_freq10 \
    --batch-size 256 --hardness 0.95 \
    --critic-weight-decay 0.00001 --critic-penalty 0.0001 --batch-norm --no-debug \
    --noise-scheduler low
"""


CUDA_VISIBLE_DEVICES=1 python train.py --algo ddpg_eagle --seed 0 --house 0 --update-freq 10 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 1000000 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/eagle_view/medium/gating_cnn/exp_low_eplen50_freq10 \
    --log-dir ./log/eagle_view/medium/gating_cnn/exp_low_eplen50_freq10 \
    --batch-size 256 --hardness 0.5 \
    --critic-weight-decay 0.00001 --critic-penalty 0.0001 --batch-norm --no-debug \
    --noise-scheduler low
