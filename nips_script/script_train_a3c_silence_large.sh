#!/bin/bash

# max steps = 80
# true batch size: 128 * 60
# silence steps = 10, gamma = 0.99
# entropy bonus = 0.02
# gradient clip = 2

CUDA_VISIBLE_DEVICES=3,4,5,6,7 python3 zmq_train.py --job-name large \
    --seed 0 --env-set train --rew-clip \
    --n-house 200 --n-proc 200 --batch-size 32 --t-max 60 --grad-batch 4 \
    --max-episode-len 80 \
    --hardness 0.95 --max-birthplace-steps 20 --min-birthplace-grids 2 \
    --reward-type new --success-measure see --reward-silence 10 \
    --multi-target --use-target-gating --include-object-target \
    --segmentation-input color --depth-input --resolution normal \
    --render-gpu 0,1,2 --max-iters 100000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.99 --batch-norm \
    --entropy-penalty 0.02 --q-loss-coef 1.0 --grad-clip 2.0 \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 20 --save-rate 1000 --eval-rate 2000000 \
    --save-dir ./_model_/nips/large/birth20_sil10_seg/a3c_bc128_gate_tmax60_et002 \
    --log-dir ./log/nips/large/birth20_sil10_seg/a3c_bc128_gate_tmax60_et002
