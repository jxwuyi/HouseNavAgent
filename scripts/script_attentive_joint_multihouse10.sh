#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python3 train.py --algo ddpg_joint --model attentive_cnn --seed 0 --update-freq 10 \
    --house -10 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 500000 \
    --lrate 0.0001 --gamma 0.95 \
    --save-dir ./_model_/att_joint/multi_house/linear_reward/medium/house10_eplen50_freq10 \
    --log-dir ./log/att_joint/multi_house/linear_reward/medium/house10_eplen50_freq10 \
    --batch-size 128 --hardness 0.6 \
    --weight-decay 0.00001 --critic-penalty 0.0001 --entropy-penalty 0.001 \
    --batch-norm --no-debug \
    --noise-scheduler high --q-loss-coef 100 --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 1 --depth-input \
    --att-resolution low --att-skip-depth
