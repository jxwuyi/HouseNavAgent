#!/bin/bash


CUDA_VISIBLE_DEVICES=6 python train.py --algo ddpg_alter --seed 0 --house 0 --update-freq 10 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 1000000 \
    --lrate 0.0001 --gamma 0.95 --entropy-penalty 0.001 \
    --save-dir ./_model_/joint/house0/linear_reward/medium/alter_lr1e4_eplen50_freq10 \
    --log-dir ./log/joint/house0/linear_reward/medium/alter_lr1e4_eplen50_freq10 \
    --batch-size 256 --hardness 0.5 \
    --noise-scheduler low \
    --weight-decay 0.00001 --critic-penalty 0.0001 --batch-norm --no-debug

  #--entropy-penalty 0.001 \
