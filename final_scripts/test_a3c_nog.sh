#!/bin/bash


"""
Test A3C + Multi-Target + Segment Color + No-Gating
"""
modelfile='./_model_/cluster/a3c_seg_nog/ZMQA3CTrainer_best.pkl'
logfile='./log/eval/cluster/a3c_seg_nog'

CUDA_VISIBLE_DEVICES=7 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input color --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

