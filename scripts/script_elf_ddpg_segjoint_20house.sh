#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python elf_train.py --seed 0 --algo ddpg --linear-reward --max-episode-len 50 \
    --render-gpu "1,2,3,4,5,6" \
    --house -20 \
    --lrate 0.0001 --gamma 0.95 \
    --save-dir ./_model_/elf/ddpg/segjoint_10house_medium/ddpg_joint_100_hist_3_c \
    --log-dir ./log/elf/ddpg/segjoint_10house_medium/ddpg_joint_100_hist_3_c \
    --hardness 0.6 \
    --weight-decay 0.0001 --critic-penalty 0.0001 --entropy-penalty 0.01 \
    --batch-norm --no-debug \
    --q-loss-coef 100 --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 \
    --num-games 40 --env-group-size 20 --batch-size 16 --elf-T 4
     #--noise-scheduler high
