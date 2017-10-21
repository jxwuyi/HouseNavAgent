#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python3 train.py --seed 0 --env-set small \
    --algo ddpg_joint --update-freq 3 --max-episode-len 100 \
    --house -20 --reward-type delta --success-measure see \
    --multi-target \
    --lrate 0.0001 --gamma 0.95 \
    --batch-size 128 --hardness 0.95 --replay-buffer-size 1000000 \
    --weight-decay 0.00001 --critic-penalty 0.0001 --entropy-penalty 0.01 \
    --batch-norm --no-debug \
    --noise-scheduler high --q-loss-coef 100 \
    --segmentation-input none --depth-input --resolution normal --history-frame-len 4 \
    --save-dir ./_model_/new/ddpg/small/hard_visual/ddpg_joint_hist_4 \
    --log-dir ./log/new/ddpg/small/hard_visual/ddpg_joint_hist_4

