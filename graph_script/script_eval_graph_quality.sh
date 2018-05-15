#!/usr/bin/env bash

MODEL_DIR="./_graph_/mix_motion/nips_tune_old/seg_mask/motion_dict.json"

NAME="tune_old_seg_mask"

FAKE_GRAPH_DIR="./_graph_/fake/mle_fake_graph_params.pkl"
RAND300_GRAPH_DIR="./_graph_/random_300/mle_random_graph_params.pkl"
RAND1000_GRAPH_DIR="./_graph_/random_1000/mle_random_graph_params.pkl"

for graph in "rand300" # "rand1000" "fake"
do
    GRAPH_DIR=$FAKE_GRAPH_DIR
    if [ $graph == "rand300" ]
    then
        GRAPH_DIR=$RAND300_GRAPH_DIR
    fi
    if [ $graph == "rand1000" ]
    then
        GRAPH_DIR=$RAND1000_GRAPH_DIR
    fi
    exp_len="50"
    for ep_len in "300" "500" "700"
    do
        python3 HRL/eval_HRL.py --seed 0 --env-set test --house -50 \
            --hardness 0.95 --render-gpu 1 --max-birthplace-steps 40 --min-birthplace-grids 1 \
            --planner graph --planner-file $GRAPH_DIR \
            --success-measure see --multi-target --use-target-gating \
            --only-eval-room-target \
            --motion mixture --mixture-motion-dict $MODEL_DIR \
            --max-episode-len $ep_len --n-exp-steps $exp_len --max-iters 5000 \
            --segmentation-input color \
            --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
            --store-history \
            --log-dir ./log/HRL/eval/eval_graph/"$graph"_graph/room_g_"$ep_len"_mix_"$exp_len"
    done
done

