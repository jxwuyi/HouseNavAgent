#!/usr/bin/env bash

motion_file="./results/motion_dict/nips_old_motion.json"
max_birth=40
name='lp1e2_seg'
flag_min_birth="--min-birthplace-grids 1"
birth_name="birth_away"
seed=5000
required="4:253,5:436"
for ep_len in "500" "300" "1000"
do
    CUDA_VISIBLE_DEVICES=1 python3 HRL/eval_motion.py --task-name roomnav --env-set test \
        --house -50 --seed $seed --render-gpu 0 \
        --max-birthplace-steps $max_birth $flag_min_birth \
        --hardness 0.95 \
        --segmentation-input color \
        --success-measure see --multi-target \
        --motion mixture --mixture-motion-dict $motion_file \
        --max-episode-len $ep_len --plan-dist-iters $required \
        --only-eval-room-target \
        --store-history \
        --log-dir ./results/additional/pure_policy/"$birth_name"_maxbth_"$max_birth"_eplen_"$ep_len"
done

#--target-mask-input

