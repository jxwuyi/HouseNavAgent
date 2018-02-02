#!/bin/bash


"""
Test NOP
"""

logfile='./log/eval/cluster/small_nop-cont'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --env-set small --house -20 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target \
  --algo nop --max-episode-len 100 --max-iters 1000 \
  --segmentation-input color \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile" --batch-norm

