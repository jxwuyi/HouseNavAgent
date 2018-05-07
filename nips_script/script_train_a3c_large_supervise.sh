#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 zmq_train.py --seed 0 --env-set train \
    --job-name sup_large --n-house 200 --n-proc 200 --batch-size 64 --t-max 30 --grad-batch 1 \
    --max-episode-len 60 --rew-clip \
    --supervised-learning \
    --hardness 0.95 --max-birthplace-steps 15 --min-birthplace-grids 2 \
    --reward-type new --success-measure see \
    --multi-target --use-target-gating --include-object-target \
    --segmentation-input color --depth-input --resolution normal \
    --render-gpu 1,2,3,4,5 --max-iters 100000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.99 \
    --entropy-penalty 0.1 --q-loss-coef 1.0 --grad-clip 1.0 --adv-norm \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 20 --save-rate 1000 --eval-rate 500000 \
    --curriculum-schedule 2,2,10000 \
    --save-dir ./_model_/nips/sup_large/birth15_seg_rewclip_nobc_curr_fine/ \
    --log-dir ./log/nips/sup_large/birth15_seg_rewclip_nobc_curr_fine/
