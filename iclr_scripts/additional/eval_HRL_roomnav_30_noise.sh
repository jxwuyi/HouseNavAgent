#!/usr/bin/env bash

MODEL_DIR="./results/motion_dict/nips_old_motion.json"

NAME="tune_old_seg_mask"

GRAPH_DIR="/home/jxwuyi/backup/HouseNavAgent/_graph_/random_300/mle_random_graph_params.pkl"

noise="0.85"

seed=7000   #5000

required="4:253,5:436"

for TERM in mask # see
do
    exp_len="30"
    for ep_len in "500" "300"
    do
        CUDA_VISIBLE_DEVICES=1 python3 HRL/eval_HRL.py --seed $seed --env-set test --house -50 \
            --hardness 0.95 --render-gpu 0 --max-birthplace-steps 50 --min-birthplace-grids 1 \
            --planner graph --planner-file $GRAPH_DIR \
            --success-measure see --multi-target --use-target-gating --terminate-measure $TERM \
            --only-eval-room-target \
            --planner-obs-noise $noise \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --plan-dist-iters $required \
            --segmentation-input color \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --store-history \
            --log-dir ./results/additional/HRL/new2_g_"$ep_len"_m_"$exp_len"_term_"$TERM"
    done
done
# --max-birthplace-steps 15
# --only-eval-object-target

# eval baseline method
# 300 steps: 26 succ, 30 reach
# 500 steps: 29.9 succ, 33 reach
