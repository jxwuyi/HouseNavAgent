#!/usr/bin/env bash

# NOTE:
#   The only difference between this script and the normal eval_BRM.sh script
#      is that here the option <--force-semantic-done> is on!
#   So the agent will terminate the episode on its own.

MODEL_DIR="./release/metadata/motion_dict.json"
SEMANTIC_DIR="./release/metadata/semantic_oracle_rooms.json"
GRAPH_DIR="./release/graph/mle_random_graph_params.pkl"

noise="0.85"

all_ep_len="1000 300"
all_exp_len="10"

seed=7
max_iters=6000

TERM="mask"

required="4:253,5:436"

for exp_len in $all_exp_len
do
    for ep_len in $all_ep_len
    do
        CUDA_VISIBLE_DEVICES=1 python3 HRL/eval_HRL.py --seed $seed --env-set test --house -50 \
            --hardness 0.95 --render-gpu 0 --max-birthplace-steps 40 --min-birthplace-grids 1 \
            --planner graph --planner-file $GRAPH_DIR \
            --success-measure see --multi-target --use-target-gating --terminate-measure $TERM \
            --only-eval-room-target \
            --planner-obs-noise $noise \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --max-iters $max_iters \
            --segmentation-input color --depth-input \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --force-semantic-done \
            --store-history \
            --log-dir ./results/terminate/BRM_main \
            --semantic-dir $SEMANTIC_DIR \
            --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0 \
            --backup-rate 1000

        # additional episodes for faraway targets
        CUDA_VISIBLE_DEVICES=1 python3 HRL/eval_HRL.py --seed 7000 --env-set test --house -50 \
            --hardness 0.95 --render-gpu 0 --max-birthplace-steps 40 --min-birthplace-grids 1 \
            --planner graph --planner-file $GRAPH_DIR \
            --success-measure see --multi-target --use-target-gating --terminate-measure $TERM \
            --only-eval-room-target \
            --planner-obs-noise $noise \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --plan-dist-iters $required \
            --segmentation-input color --depth-input \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --force-semantic-done \
            --store-history \
            --log-dir ./results/terminate/BRM_add \
            --semantic-dir $SEMANTIC_DIR \
            --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0 \
            --backup-rate 1000
    
    done
done
