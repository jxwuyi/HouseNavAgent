#!/usr/bin/env bash

MODEL_DIR="./release/metadata/motion_dict.json"
seed=7
required="4:253,5:436"

for ep_len in "300" "1000"
do
    CUDA_VISIBLE_DEVICES=2 python3 HRL/eval_motion.py --task-name roomnav --env-set test \
        --house -50 --seed $seed --render-gpu 1 \
        --max-birthplace-steps 40 --min-birthplace-grids 1 \
        --hardness 0.95 \
        --segmentation-input color --depth-input \
        --success-measure see --multi-target \
        --motion mixture --mixture-motion-dict $MODEL_DIR \
        --max-episode-len $ep_len --max-iters 5000 \
        --only-eval-room-target \
        --store-history \
        --log-dir ./results/pure_policy_main
    
    # additional episodes for faraway targets
    CUDA_VISIBLE_DEVICES=2 python3 HRL/eval_motion.py --task-name roomnav --env-set test \
        --house -50 --seed 7000 --render-gpu 1 \
        --max-birthplace-steps 40 --min-birthplace-grids 1 \
        --hardness 0.95 \
        --segmentation-input color --depth-input \
        --success-measure see --multi-target \
        --motion mixture --mixture-motion-dict $MODEL_DIR \
        --max-episode-len $ep_len --plan-dist-iters $required \
        --only-eval-room-target \
        --store-history \
        --log-dir ./results/pure_policy_add
done


