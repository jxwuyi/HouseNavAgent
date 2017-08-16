#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python train.py --seed 0 --algo ddpg_joint --update-freq 10 --linear-reward --max-episode-len 50 \
    --house -3 \
    --lrate 0.0001 --gamma 0.95 \
    --save-dir ./_model_/multi_house/linear_reward/medium/ddpg_joint_150_exp_high \
    --log-dir ./log/multi_house/linear_reward/medium/ddpg_joint_150_exp_high \
    --batch-size 256 --hardness 0.6 --replay-buffer-size 1000000 \
    --weight-decay 0.00001 --critic-penalty 0.0001 --entropy-penalty 0.001 \
    --batch-norm --no-debug \
    --noise-scheduler high --q-loss-coef 150
