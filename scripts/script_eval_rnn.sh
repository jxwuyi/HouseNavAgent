#!/bin/bash

modelfile='./_model_/rnn/linear_reward/hard/rdpg_cnn_critic/RDPG_RDPGTrainer_best.pkl'
logfile='./log/eval/rdpg/noise/hard/defaults'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 0 --seed 211 --hardness 0.95 --algo rdpg \
    --max-episode-len 150 --max-iters 100 --store-history \
    --log-dir "$logfile"  --warmstart "$modelfile" \
    --rnn-cell lstm --rnn-units 100 --rnn-layers 1
