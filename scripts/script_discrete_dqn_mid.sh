#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python train.py --algo dqn \
    --seed 0 --house 0 --update-freq 10 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 1000000 \
    --lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/discrete/dqn/medium/eplen50_freq10_exp_low \
    --log-dir ./log/discrete/dqn/medium/eplen50_freq10_exp_low \
    --batch-size 256 --hardness 0.5 \
    --noise-scheduler medium \
    --weight-decay 0.0001 \
    --batch-norm --no-debug  # --critic-penalty 0.0001
