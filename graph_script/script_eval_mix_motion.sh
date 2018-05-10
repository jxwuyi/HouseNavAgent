#!/usr/bin/env bash

python3 HRL/eval_motion.py --task-name roomnav --env-set test \
    --house -50 --seed 0 \
    --render-gpu 6 --hardness 0.95 \
    --segmentation-input none --depth-input \
    --success-measure see --multi-target --include-object-target \
    --motion mixture --mixture-motion-dict ./_graph_/mix_motion/visual_seg_mask/motion_dict.json \
    --max-episode-len 100 --max-iters 5000 \
    --only-eval-room-target \
    --store-history \
    --log-dir ./log/graph/eval/mix_visual_room_100

# --target-mask-input

