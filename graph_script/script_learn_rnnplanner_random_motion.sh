#!/usr/bin/env bash

python3 HRL/learn_controller.py --task-name roomnav --env-set train --house -200 --seed 0 \
    --render-gpu 0 \
    --segmentation-input color \
    --success-measure see \
    --include-object-target --only-eval-object-target \
    --units 50 --iters 10000 \
    --max-episode-len 300 --max-exp-steps 50 --max-planner-steps 10 \
    --batch-size 64 --lrate 0.001 --weight-decay 0.00001 --grad-clip 5 \
    --entropy-penalty 0.01 --gamma 0.99 \
    --time-penalty 0.1 --success-reward 2 \
    --motion random --random-motion-skill 6 --terminate-measure mask \
    --save-dir ./_graph_/controller/p300_rand50_skill6
