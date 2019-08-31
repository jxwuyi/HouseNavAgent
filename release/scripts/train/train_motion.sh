#!/bin/bash

# change target to the desired the semantic goal to train the conditional policy
target="kitchen"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 zmq_train.py --job-name large \
    --fixed-target $target \
    --seed 0 --env-set train --rew-clip 3 \
    --n-house 200 --n-proc 200 --batch-size 64 --t-max 30 --grad-batch 1  \
    --max-episode-len 60 \
    --curriculum-schedule 5,3,10000 \
    --hardness 0.95 --max-birthplace-steps 15 --min-birthplace-grids 2 \
    --reward-type new --success-measure see \
    --multi-target --use-target-gating \
    --segmentation-input color --depth-input --resolution normal \
    --render-gpu 1,2,3,4,5 --max-iters 100000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.97 --batch-norm \
    --entropy-penalty 0.1 --logits-penalty 0.01 --q-loss-coef 1.0 --grad-clip 1.0 --adv-norm \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 20 --save-rate 1000 --eval-rate 200000 \
    --save-dir ./model/$target \
    --log-dir ./log/$target
