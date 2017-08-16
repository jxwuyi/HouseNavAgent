#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python train.py --seed 0 --house 0 --update-freq 10 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 1000000 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/large_net/house0/linear_reward/medium/bc_eplen50_freq10 \
    --log-dir ./log/large_net/house0/linear_reward/medium/bc_eplen50_freq10 \
    --batch-size 256 --hardness 0.5 \
    --noise-scheduler medium \
    --critic-weight-decay 0.00001 --critic-penalty 0.0001 --batch-norm --no-debug

  #--entropy-penalty 0.001 \
