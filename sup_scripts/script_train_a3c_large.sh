#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python3 zmq_train.py --job-name large \
    --seed 0 --env-set train --rew-clip 3 \
    --n-house 200 --n-proc 200 --batch-size 64 --t-max 30 --grad-batch 1  \
    --max-episode-len 60 \
    --hardness 0.6 --max-birthplace-steps 15 --min-birthplace-grids 2 \
    --reward-type new --success-measure see \
    --multi-target --use-target-gating --fixed-target kitchen \
    --segmentation-input color --depth-input --resolution normal \
    --render-gpu 0,1 --train-gpu 0 --max-iters 50000 \
    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.97 --batch-norm \
    --entropy-penalty 0.1 --logits-penalty 0.01 --q-loss-coef 1.0 --grad-clip 1.0 --adv-norm \
    --curriculum-schedule 5,3,10000 \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --report-rate 20 --save-rate 1000 --eval-rate 1000000 \
    --save-dir ./_model_/sup_tune/kitchen/a3c_bc64_gate_tmax30 \
    --log-dir ./log/sup_tune/kitchen/a3c_bc64_gate_tmax30 \
    --warmstart ./_model_/supervise/train_kitchen_t20_ent5e1_lgt1e1/SUP_best.pkl 
