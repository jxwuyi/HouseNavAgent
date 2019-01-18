#!/usr/bin/env bash

SEMANTIC_DIR="./_model_/semantic/_dict_/semantic_oracle_rooms.json"
motion_file="./results/motion_dict/nips_old_motion.json"
max_birth=40
name='lp1e2_seg'
flag_min_birth="--min-birthplace-grids 1"
birth_name="birth_away"
#seed="7"
seed=0
max_iters="5000" #"10000"
for ep_len in "300" "500" "1000"
do
    CUDA_VISIBLE_DEVICES=2 python3 HRL/eval_motion.py --task-name roomnav --env-set test \
        --house -50 --seed $seed --render-gpu 0 \
        --max-birthplace-steps $max_birth $flag_min_birth \
        --hardness 0.95 \
        --segmentation-input color --depth-input \
        --success-measure see --multi-target \
        --motion mixture --mixture-motion-dict $motion_file \
        --max-episode-len $ep_len --max-iters $max_iters \
        --only-eval-room-target \
        --store-history \
        --log-dir ./results/force_terminate/pure_policy/"$birth_name"_maxbth_"$max_birth"_eplen_"$ep_len" \
        --force-semantic-done \
        --semantic-dir $SEMANTIC_DIR \
        --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0
done

#--target-mask-input

