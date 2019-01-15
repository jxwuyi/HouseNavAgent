#!/usr/bin/env bash

MODEL_DIR="./results/motion_dict/mask_feature_motion.json"
SEMANTIC_DIR="./_model_/semantic/_dict_/semantic_oracle_rooms.json"
max_birth="40"
#seed=7
seed=0
#max_iters="10000"
max_iters=5000
for ep_len in "300" "500" "1000"    #"500" "1000" "300"
do
    CUDA_VISIBLE_DEVICES=2 python3 HRL/eval_motion.py --task-name roomnav --env-set test \
        --house -50 --seed $seed \
        --render-gpu 1 --hardness 0.95 \
        --max-birthplace-steps $max_birth --min-birthplace-grids 1 \
        --segmentation-input color --depth-input \
        --success-measure see --multi-target \
        --motion mixture --mixture-motion-dict $MODEL_DIR \
        --max-episode-len $ep_len --max-iters $max_iters \
        --only-eval-room-target \
        --store-history \
        --log-dir ./results/force_terminate/mask_feature/maxbth_"$max_birth"_eplen_"$ep_len" \
        --force-semantic-done \
        --semantic-dir $SEMANTIC_DIR \
        --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0
done

