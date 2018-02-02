#!/bin/bash


"""
Test A3C + Multi-Target + Segment-Color + Gating + Tmax30
"""
#modelfile='./_model_/cluster/full/a3c_seg_gate/ZMQA3CTrainer.pkl'
#logfile='./log/eval/cluster/full/a3c_seg_gate_final'

#modelfile='./_model_/cluster/full/a3c_seg_nog_ent0075/ZMQA3CTrainer_succ.pkl'
#logfile='./log/eval/cluster/full/a3c_seg_nog_ent0075_succ'

modelfile='./_model_/cluster/full/a3c_vis_nog_ent005/ZMQA3CTrainer_best.pkl'
logfile='./log/eval/cluster/full/a3c_vis_nog_ent005_best'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input none --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

