#!/usr/bin/env bash

MODEL_DIR="./release/metadata/motion_dict.json"

EP_LEN="500"
EXP_LEN="50"
MAX_OPT="10"

python3 HRL/learn_controller.py --task-name roomnav --env-set train --house -200 --seed 0 \
    --render-gpu 0 \
    --segmentation-input color \
    --success-measure see \
    --only-eval-room-target\
    --units 50 --iters 50000 \
    --max-episode-len $EP_LEN --max-exp-steps $EXP_LEN --max-planner-steps $MAX_OPT \
    --batch-size 32 --lrate 0.001 --weight-decay 0.00001 --grad-clip 5 \
    --entropy-penalty 0.01 --gamma 0.99 \
    --time-penalty 0.1 --success-reward 2 \
    --motion mixture --mixture-motion-dict $MODEL_DIR --terminate-measure mask \
    --save-dir ./graph/rnn_controller
