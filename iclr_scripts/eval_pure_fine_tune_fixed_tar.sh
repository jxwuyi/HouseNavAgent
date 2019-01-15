#!/usr/bin/env bash

target='kitchen'
policy='./_model_/sup_tune/kitchen/a3c_bc64_gate_tmax30/ZMQA3CTrainer_final.pkl'
policy_args='./_model_/sup_tune/kitchen/a3c_bc64_gate_tmax30/train_args.json'

CUDA_VISIBLE_DEVICES=2 python3 HRL/eval_motion.py --task-name roomnav --env-set test --seed 0 \
    --house -50 --seed 0 \
    --render-gpu 1 --hardness 0.95 \
    --max-birthplace-steps 40 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input \
    --success-measure see --multi-target \
    --motion rnn \
    --max-episode-len 100 --max-iters 5000 \
    --fixed-target $target \
    --store-history \
    --log-dir ./log/iclr/eval_pure_new_$target \
    --warmstart $policy \
    --warmstart-dict $policy_args
