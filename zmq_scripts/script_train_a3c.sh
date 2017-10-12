#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7 python3 zmq_train.py --seed 0 --env-set train \
    --n-house 200 --n-proc 200 --batch-size 64 --t-max 5 --max-episode-len 100 \
    --hardness 0.9 --reward-type delta --multi-target --use-target-gating \
    --segmentation-input color --depth-input --resolution normal \
    --max-iters 600000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.95 --batch-norm \
    --entropy-penalty 0.1 --q-loss-coef 1.0 --grad-clip 0.5 \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 20 --save-rate 1000 --eval-rate 100000 \
    --save-dir ./_model_/zmq_a3c/train_multitarget/medium_color/bn_bc64_tmax5_gate \
    --log-dir ./log/zmq_a3c/train_multitarget/medium_color/bn_bc64_tmax5_gate 

