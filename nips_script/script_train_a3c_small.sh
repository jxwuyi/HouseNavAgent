#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python3 zmq_train.py --seed 0 --env-set small \
    --n-house 20 --n-proc 90 --batch-size 64 --t-max 30 --max-episode-len 100 \
    --hardness 0.95 --max-birthplace-steps 20 \
    --reward-type new --success-measure see \
    --multi-target --use-target-gating --include-object-target \
    --segmentation-input color --depth-input --resolution normal \
    --render-gpu 0,1,2 --max-iters 100000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.95 --batch-norm \
    --entropy-penalty 0.1 --q-loss-coef 1.0 --grad-clip 1.0 \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 20 --save-rate 1000 --eval-rate 200000 \
    --save-dir ./_model_/nips/small/birth20_seg/a3c_bc64_gate_tmax30 \
    --log-dir ./log/nips/small/birth20_seg/a3c_bc64_gate_tmax30
