#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python train.py --algo dqn \
    --seed 0 --house 0 --update-freq 10 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 1000000 \
    --lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/discrete/dqn/medium/eplen50_freq10_exp_exp \
    --log-dir ./log/discrete/dqn/medium/eplen50_freq10_exp_exp \
    --batch-size 256 --hardness 0.5 \
    --noise-scheduler exp \
    --weight-decay 0.00001 \
    --batch-norm --no-debug  # --critic-penalty 0.0001
