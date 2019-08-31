#!/usr/bin/env bash


MODEL_DIR="./release/metadata/motion_dict.json"
SEMANTIC_DIR="./release/metadata/semantic_oracle_rooms.json"
GRAPH_DIR="./release/rnn_controller/RNNPlanner.pkl"

all_exp_len="10"

all_ep_len="300 1000"
max_iters="6000"

seed="7"

required="4:253,5:436"

for exp_len in $all_exp_len
do
    for ep_len in $all_ep_len
    do
       CUDA_VISIBLE_DEVICES=0 python3 HRL/eval_HRL.py --seed $seed --env-set test --house -50 \
            --hardness 0.95 --render-gpu 2 --max-birthplace-steps 40 --min-birthplace-grids 1 \
            --planner rnn --planner-file $GRAPH_DIR --planner-units 50 \
            --success-measure see --multi-target --use-target-gating \
            --terminate-measure mask \
            --only-eval-room-target \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --max-iters $max_iters \
            --segmentation-input color --depth-input \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --store-history \
            --log-dir ./results/RNN_control_main \
            --semantic-dir $SEMANTIC_DIR \
            --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0 \
            --backup-rate 1000
        
        # additional episodes for faraway targets
        CUDA_VISIBLE_DEVICES=0 python3 HRL/eval_HRL.py --seed 7000 --env-set test --house -50 \
            --hardness 0.95 --render-gpu 2 --max-birthplace-steps 40 --min-birthplace-grids 1 \
            --planner rnn --planner-file $GRAPH_DIR --planner-units 50 \
            --success-measure see --multi-target --use-target-gating \
            --terminate-measure mask \
            --only-eval-room-target \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --plan-dist-iters $required \
            --segmentation-input color --depth-input \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --store-history \
            --log-dir ./results/RNN_control_add \
            --semantic-dir $SEMANTIC_DIR \
            --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0
    done
done

