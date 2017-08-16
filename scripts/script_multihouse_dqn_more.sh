#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python train.py --seed 0 --algo dqn --update-freq 10 --linear-reward --max-episode-len 60 \
    --house -6 \
    --lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/multi_house/linear_reward/6house_medium/dqn_eplen60_exp_exp \
    --log-dir ./log/multi_house/linear_reward/6house_medium/dqn_eplen60_exp_exp \
    --batch-size 256 --hardness 0.6 --replay-buffer-size 1000000 \
    --weight-decay 0.00001 \
    --batch-norm --no-debug \
    --noise-scheduler exp
