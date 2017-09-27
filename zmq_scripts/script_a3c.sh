#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python3 zmq_train.py --seed 0 \
    --n-house 5 --n-proc 5 --batch-size 5 --t-max 10 --max-episode-len 50 \
    --hardness 0.6 --segmentation-input joint --depth-input \
    --train-gpu 1 --render-gpu 0 --max-iters 100000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.95 --batch-norm \
    --entropy-penalty 0.01 --q-loss-coef 1.0 --grad-clip 2.0 \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 1 --save-rate 100 --eval-rate 1000 \
    --save-dir ./_model_/zmq_a3c/5house/medium/bn_bc5_tmax10 \
    --log-dir ./log/zmq_a3c/5house/medium/bn_bc5_tmax10

