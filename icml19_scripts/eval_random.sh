#!/usr/bin/env bash

SEMANTIC_DIR="./_model_/semantic/_dict_/semantic_oracle_rooms.json"
max_birth=40
seed=0
max_iters=5000
for task in "room_nav" # "obj_nav"
do
    flag_object="--include-object-target"
    flag_target="--only-eval-object-target"
    if [ $task == "room_nav" ]
    then
        flag_object=""
        flag_target="--only-eval-room-target"
    fi
    for ep_len in "300" #"500" "1000"
    do
        CUDA_VISIBLE_DEVICES=1 python3 HRL/eval_motion.py --task-name roomnav --env-set test \
            --house -50 --seed $seed \
            --max-birthplace-steps $max_birth --min-birthplace-grids 1 \
            --render-gpu 1 --hardness 0.95 \
            --segmentation-input color --depth-input \
            --success-measure see --multi-target $flag_object $flag_target \
            --motion random \
            --max-episode-len $ep_len --max-iters $max_iters \
            --store-history \
            --log-dir ./results/force_terminate/random/maxbth_"$max_birth"_eplen_"$ep_len" \
            --force-semantic-done \
            --semantic-dir $SEMANTIC_DIR \
            --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0
    done
done

#--target-mask-input

