#!/usr/bin/env bash

MODEL_DIR="./results/motion_dict/nips_old_motion.json"

SEMANTIC_DIR="./_model_/semantic/_dict_/semantic_oracle_rooms.json"

noise="0.85"

all_ep_len="1000"
#all_ep_len="300 500 1000"

all_exp_len="10 30 50"

#seed=0
seed=7
max_iters=10000

#seed=7
#max_iters="10000"

TERM="mask"

for exp_len in $all_exp_len
do
    for ep_len in $all_ep_len
    do
        CUDA_VISIBLE_DEVICES=1 python3 HRL/eval_HRL.py --seed $seed --env-set test --house -50 \
            --hardness 0.95 --render-gpu 1 --max-birthplace-steps 40 --min-birthplace-grids 1 \
            --planner oracle \
            --success-measure see --multi-target --use-target-gating --terminate-measure $TERM \
            --only-eval-room-target \
            --planner-obs-noise $noise \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --max-iters $max_iters \
            --segmentation-input color --depth-input \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --store-history \
            --log-dir ./results/iccv/HRL_oracle/g_"$ep_len"_m_"$exp_len"_term_"$TERM"_sd"$seed"
    done
done
# --max-birthplace-steps 15
# --only-eval-object-target

# eval baseline method
# 300 steps: 26 succ, 30 reach
# 500 steps: 29.9 succ, 33 reach
