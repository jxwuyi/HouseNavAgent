#!/bin/bash


"""
Test A3C + NO-Gating + Multi-Target + Segment + Tmax30
"""
#modelfile='./_model_/cluster/small/a3c_segcolor_nog/ZMQA3CTrainer_succ.pkl'
#logfile='./log/cluster/small/small_a3c_segcolor_nog'
modelfile='./_model_/cluster/full/a3c_segcolor_nog_ent005/ZMQA3CTrainer_succ.pkl'
logfile='./log/eval/cluster/full/full_a3c_segcolor_nog_ent005'

CUDA_VISIBLE_DEVICES=2 python3 eval.py --env-set train --house -200 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input color --depth-input  \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

