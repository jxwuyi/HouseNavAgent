#!/usr/bin/env bash

max_birth=40
seed=5000
required="4:253,5:436"
for task in "room_nav" # "obj_nav"
do
    flag_object="--include-object-target"
    flag_target="--only-eval-object-target"
    if [ $task == "room_nav" ]
    then
        flag_object=""
        flag_target="--only-eval-room-target"
    fi
    for ep_len in "500" "1000" "300"
    do
        python3 HRL/eval_motion.py --task-name roomnav --env-set test \
            --house -50 --seed $seed \
            --max-birthplace-steps $max_birth --min-birthplace-grids 1 \
            --render-gpu 1 --hardness 0.95 \
            --segmentation-input color \
            --success-measure see --multi-target $flag_object $flag_target \
            --motion random \
            --max-episode-len $ep_len --plan-dist-iters $required \
            --store-history \
            --log-dir ./results/additional/random/maxbth_"$max_birth"_eplen_"$ep_len"
    done
done

#--target-mask-input

