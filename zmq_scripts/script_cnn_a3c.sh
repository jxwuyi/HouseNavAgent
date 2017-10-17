#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 python3 zmq_train.py --seed 0 \
    --n-house 1 --n-proc 50 --batch-size 32 --t-max 5 --max-episode-len 50 \
    --hardness 0.6 --reward-type delta \
    --segmentation-input joint --depth-input --resolution normal \
    --render-gpu 0,1,2 --max-iters 100000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.95 --batch-norm \
    --entropy-penalty 0.1 --q-loss-coef 1.0 --grad-clip 0.5 \
    --feed-forward-a3c --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 20 --save-rate 1000 --eval-rate 100000 \
    --save-dir ./_model_/zmq_cnn_a3c/1house/medium/bn_bc32_tmax5 \
    --log-dir ./log/zmq_cnn_a3c/1house/medium/bn_bc32_tmax5

