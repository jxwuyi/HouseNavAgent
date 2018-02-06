#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python3 train.py --seed 0 --env-set small --render-gpu 1 \
    --algo ddpg_joint --update-freq 10 --max-episode-len 100 \
    --house -20 --reward-type delta --success-measure see \
    --multi-target --use-target-gating --use-action-gating --include-object-target \
    --lrate 0.0001 --gamma 0.95 \
    --batch-size 128 --hardness 0.95 --replay-buffer-size 200000 \
    --weight-decay 0.00001 --critic-penalty 0.0001 --entropy-penalty 0.01 \
    --batch-norm --no-debug \
    --noise-scheduler high --q-loss-coef 100 \
    --segmentation-input color --depth-input --resolution normal --history-frame-len 4 \
    --save-dir ./_model_/icml/small_object/hard_seg/ddpg_gate_hist_4 \
    --log-dir ./log/icml/small_object/hard_seg/ddpg_gate_hist_4

