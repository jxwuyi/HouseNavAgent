#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python train.py --algo ddpg_eagle --seed 0 --house -3 --update-freq 10 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 1000000 \
    --lrate 0.0001 --critic-lrate 0.001 --gamma 0.95 \
    --save-dir ./_model_/eagle_view/medium/gate_resnet/exp_high_eplen50_freq10 \
    --log-dir ./log/eagle_view/medium/gate_resnet/exp_high_eplen50_freq10 \
    --batch-size 256 --hardness 0.6 \
    --critic-weight-decay 0.0001 --critic-penalty 0.0001 --batch-norm --no-debug \
    --noise-scheduler high --use-residual-critic #--dist-sampling
