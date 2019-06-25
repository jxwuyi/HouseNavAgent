#!/usr/bin/env bash

MODEL_DIR="./results/motion_dict/mask_feature_motion.json"
max_birth="40"
seed=5000
required="4:253,5:436"
for ep_len in "500" "1000" "300"
do
    CUDA_VISIBLE_DEVICES=2 python3 HRL/eval_motion.py --task-name roomnav --env-set test \
        --house -50 --seed $seed \
        --render-gpu 1 --hardness 0.95 \
        --max-birthplace-steps $max_birth --min-birthplace-grids 1 \
        --segmentation-input color \
        --success-measure see --multi-target \
        --motion mixture --mixture-motion-dict $MODEL_DIR \
        --max-episode-len $ep_len --plan-dist-iters $required \
        --only-eval-room-target \
        --store-history \
        --log-dir ./results/additional/mask_feature/maxbth_"$max_birth"_eplen_"$ep_len"
done

