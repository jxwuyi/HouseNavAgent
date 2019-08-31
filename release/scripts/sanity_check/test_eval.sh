#!/usr/bin/env bash

# sanity check -- evaluation

MODEL_DIR="./release/metadata/motion_dict.json"
SEMANTIC_DIR="./release/metadata/semantic_oracle_rooms.json"
GRAPH_DIR="./release/graph/mle_random_graph_params.pkl"

noise="0.85"

all_ep_len="10"
all_exp_len="3"

seed=7
max_iters=10

TERM="mask"

for exp_len in $all_exp_len
do
    for ep_len in $all_ep_len
    do
        CUDA_VISIBLE_DEVICES=0 python3 HRL/eval_HRL.py --seed $seed --env-set test --house -2 \
            --hardness 0.95 --render-gpu 0 --max-birthplace-steps 40 --min-birthplace-grids 1 \
            --planner graph --planner-file $GRAPH_DIR \
            --success-measure see --multi-target --use-target-gating --terminate-measure $TERM \
            --only-eval-room-target \
            --planner-obs-noise $noise \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --max-iters $max_iters \
            --segmentation-input color --depth-input \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --store-history \
            --log-dir ./results/sanity \
            --semantic-dir $SEMANTIC_DIR \
            --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0 \
            --backup-rate 10
    done
done
