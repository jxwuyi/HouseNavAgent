#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python train.py --algo rdpg --seed 0 \
    --house 0 --linear-reward \
    --rnn-cell lstm --rnn-units 100 --rnn-layers 1 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/rnn/linear_reward/easy/bc_rdpg_cnn_critic \
    --log-dir ./log/rnn/linear_reward/easy/bc_rdpg_cnn_critic \
    --max-episode-len 40 --replay-buffer-size 20000 \
    --batch-size 64 --batch-length 20 --hardness 0.3 \
    --critic-weight-decay 0.0001 --critic-penalty 0.0001 --batch-norm \
    --weight-decay 0.00001 \
    --noise-scheduler low --no-debug
