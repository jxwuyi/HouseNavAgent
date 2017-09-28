#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python3 zmq_train.py --seed 0 \
    --n-house 10 --n-proc 64 --batch-size 64 --t-max 15 --max-episode-len 50 \
    --hardness 0.6 --segmentation-input joint --depth-input --resolution normal \
    --render-gpu 0,1,2 --max-iters 100000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.95 --batch-norm \
    --entropy-penalty 0.01 --q-loss-coef 1.0 --grad-clip 2.0 \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 20 --save-rate 1000 --eval-rate 100000 \
    --save-dir ./_model_/zmq_a3c/5house/medium/bn_bc64_tmax15 \
    --log-dir ./log/zmq_a3c/5house/medium/bn_bc64_tmax15

