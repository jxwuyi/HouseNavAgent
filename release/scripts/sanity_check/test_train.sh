#!/bin/bash

# sanity check of training
target="kitchen"

CUDA_VISIBLE_DEVICES=0 python3 zmq_train.py --job-name large \
    --fixed-target $target \
    --seed 0 --env-set train --rew-clip 3 \
    --n-house 2 --n-proc 2 --batch-size 2 --t-max 2 --grad-batch 1  \
    --max-episode-len 5 \
    --curriculum-schedule 5,3,1 \
    --hardness 0.95 --max-birthplace-steps 15 --min-birthplace-grids 2 \
    --reward-type new --success-measure see \
    --multi-target --use-target-gating \
    --segmentation-input color --depth-input --resolution normal \
    --render-gpu 0 --max-iters 10 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.97 --batch-norm \
    --entropy-penalty 0.1 --logits-penalty 0.01 --q-loss-coef 1.0 --grad-clip 1.0 --adv-norm \
    --rnn-units 16 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 1 --save-rate 5 --eval-rate 200000 \
    --save-dir ./model/sanity_$target \
    --log-dir ./log/sanity_$target
