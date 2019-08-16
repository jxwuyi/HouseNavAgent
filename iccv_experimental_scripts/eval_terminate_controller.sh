#!/usr/bin/env bash


MODEL_DIR="./results/motion_dict/nips_old_motion.json"
SEMANTIC_DIR="./_model_/semantic/_dict_/semantic_oracle_rooms.json"
GRAPH_DIR="/home/jxwuyi/backup/HouseNavAgent/_graph_/controller_room/mix_motion/p300_m30_max10_bc32/RNNPlanner.pkl"

#all_exp_len="10 15 20"
all_exp_len="5"

#all_ep_len="300 500 1000"
all_ep_len="1000"

max_iters="5000"   #"10000"

#seed="0"
seed="7"

TERM="mask"

for exp_len in $all_exp_len
do
    for ep_len in $all_ep_len
    do
       CUDA_VISIBLE_DEVICES=0 python3 HRL/eval_HRL.py --seed $seed --env-set test --house -50 \
            --hardness 0.95 --render-gpu 2 --max-birthplace-steps 40 --min-birthplace-grids 1 \
            --planner rnn --planner-file $GRAPH_DIR --planner-units 50 \
            --success-measure see --multi-target --use-target-gating \
            --terminate-measure $TERM \
            --only-eval-room-target \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --max-iters $max_iters \
            --segmentation-input color --depth-input \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --store-history \
            --log-dir ./results/iccv/experimental/force_terminate/controller/m_"$exp_len"_g_"$ep_len"_sd"$seed" \
            --force-semantic-done \
            --semantic-dir $SEMANTIC_DIR \
            --semantic-threshold 0.9 --semantic-filter-steps 3 --semantic-gpu 0
    done
done

# --max-birthplace-steps 15
# --only-eval-object-target

# eval baseline method
# 300 steps: 26 succ, 30 reach
# 500 steps: 29.9 succ, 33 reach
