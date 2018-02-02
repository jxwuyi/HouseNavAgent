#!/bin/bash


"""
Test A3C + Multi-Target + Segcolor + Tmax30
"""

modelfile='./_model_/cluster/full/a3c_visual_gate_ent005_sd3137/ZMQA3CTrainer_succ.pkl'
logfile='./log/eval/cluster/full/a3c_vis_gate_ent005_sd3137_succ'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target --use-target-gating \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input none --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

modelfile='./_model_/cluster/full/a3c_visual_gate_ent005_sd3137/ZMQA3CTrainer_best.pkl'
logfile='./log/eval/cluster/full/a3c_vis_gate_ent005_sd3137_best'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target --use-target-gating \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input none --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

