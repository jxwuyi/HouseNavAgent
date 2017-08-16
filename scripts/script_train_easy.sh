#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python train.py --seed 0 --update-freq 25 --linear-reward \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/linear_reward/easy/no_bc_lr1e4_1e3_sftmx_c \
    --log-dir ./log/linear_reward/easy/no_bc_lr1e4_1e3_sftmx_c \
    --batch-size 256 --hardness 0.3 \
    --critic-weight-decay 0.0001
