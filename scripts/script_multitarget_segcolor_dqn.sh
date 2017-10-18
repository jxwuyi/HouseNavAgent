#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python train.py --seed 0 --algo dqn --update-freq 5 --max-episode-len 60 \
    --house -20 --reward-type delta --success-measure see --multi-target \
    --lrate 0.0001 --gamma 0.6 \
    --save-dir ./_model_/multi_target/20house_medium/delta_segcolor_dqn_eplen60_exp_exp \
    --log-dir ./log/multi_target/20house_medium/delta_segcolor_dqn_eplen60_exp_exp \
    --batch-size 256 --hardness 0.95 --replay-buffer-size 1000000 \
    --weight-decay 0.00001 \
    --batch-norm --no-debug \
    --noise-scheduler exp \
    --segmentation-input joint --depth-input --resolution normal --history-frame-len 3
