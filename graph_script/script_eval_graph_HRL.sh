#!/usr/bin/env bash

#MODEL_DIR='./_model_/nips/large/birth15_seg_mask_curr/ZMQA3CTrainer_succ.pkl'
MODEL_DIR='./_model_/nips_new/sup_large_no_outdoor/seg_gate_bc_curr/ZMQA3CTrainer.pkl'

python3 HRL/eval_HRL.py --seed 0 --env-set test --house -50 \
    --hardness 0.95 --render-gpu 1 --no-outdoor-target \
    --planner graph --planner-file ./_graph_/fake/mle_fake_graph_params.pkl --n-exp-steps 50 \
    --success-measure see --multi-target --use-target-gating \
    --include-object-target --only-eval-object-target \
    --motion rnn --max-episode-len 300 --max-iters 5000 \
    --segmentation-input color --depth-input --no-outdoor-target \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/HRL/eval/mle_fake_graph \
    --warmstart $MODEL_DIR

# --max-birthplace-steps 15
# --only-eval-object-target

# eval baseline method
# 300 steps: 26 succ, 30 reach
# 500 steps: 29.9 succ, 33 reach
