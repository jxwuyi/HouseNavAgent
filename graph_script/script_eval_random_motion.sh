#!/usr/bin/env bash

python3 HRL/eval_motion.py --task-name roomnav --env-set test --house -50 --seed 0 \
    --render-gpu 0 --hardness 0.95 \
    --segmentation-input none --depth-input \
    --success-measure see --multi-target --include-object-target \
    --motion random --max-episode-len 100 --max-iters 5000 --store-history \
    --log-dir ./log/graph/eval/pure_fake

# --max-birthplace-steps 15
# --only-eval-object-target
