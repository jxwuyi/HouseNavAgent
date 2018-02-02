#!/bin/bash


"""
Test A3C + NO-Gating + Multi-Target + Visual + Tmax30
"""
modelfile='./_model_/cluster/small/a3c_onlyvis_nog/ZMQA3CTrainer_best.pkl'
logfile='./log/cluster/small/small_a3c_onlyvis_nog'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --env-set small --house -20 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input none \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

