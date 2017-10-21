#!/bin/bash

CUDA_VISIBLE_DEVICES=4,7 python3 train.py --seed 0 --env-set small --render-gpu 1 \
    --algo ddpg_joint --update-freq 3 --max-episode-len 100 \
    --house -20 --reward-type delta --success-measure see \
    --multi-target --use-target-gating --use-action-gating \
    --lrate 0.0001 --gamma 0.95 \
    --batch-size 128 --hardness 0.95 --replay-buffer-size 1000000 \
    --weight-decay 0.00001 --critic-penalty 0.0001 --entropy-penalty 0.01 \
    --batch-norm --no-debug \
    --noise-scheduler high --q-loss-coef 100 \
    --segmentation-input color --depth-input --resolution normal --history-frame-len 4 \
    --save-dir ./_model_/new/ddpg/small/hard_color/ddpg_joint_gate_hist_4 \
    --log-dir ./log/new/ddpg/small/hard_color/ddpg_joint_gate_hist_4

