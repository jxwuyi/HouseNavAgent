#!/usr/bin/env bash

target='kitchen'
policy='/home/jxwuyi/backup/HouseNavAgent/_model_/nips_tune_old/seg_large_c5_3_1w/kitchen/ZMQA3CTrainer.pkl'
policy_args='/home/jxwuyi/backup/HouseNavAgent/_model_/nips_tune_old/seg_large_c5_3_1w/kitchen/train_args.json'

CUDA_VISIBLE_DEVICES=1 python3 HRL/eval_motion.py --task-name roomnav --env-set test --seed 0 \
    --house -50 --seed 0 \
    --render-gpu 0 --hardness 0.95 \
    --max-birthplace-steps 40 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input \
    --success-measure see --multi-target \
    --motion rnn \
    --max-episode-len 100 --max-iters 5000 \
    --fixed-target $target \
    --store-history \
    --log-dir ./log/iclr/eval_pure_old_$target \
    --warmstart $policy \
    --warmstart-dict $policy_args
