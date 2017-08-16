#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python train.py --algo a2c \
    --seed 0 --house 0 --update-freq 10 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 50000 \
    --lrate 0.0001 --gamma 0.95 \
    --save-dir ./_model_/discrete/a2c/medium/eplen50_freq10 \
    --log-dir ./log/discrete/a2c/medium/eplen50_freq10 \
    --batch-size 256 --hardness 0.5 \
    --noise-scheduler low \
    --entropy-penalty 0.001 --batch-norm --no-debug  # --critic-penalty 0.0001
