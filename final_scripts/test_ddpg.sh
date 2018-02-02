#!/bin/bash


"""
Test DDPG + Gating + Multi-Target + SegColor
"""
modelfile='./_model_/cluster/full/ddpg_segcolor_nog/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/cluster/full/ddpg_segcolor_nog'

CUDA_VISIBLE_DEVICES=1,2 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.95 --render-gpu 1 \
  --success-measure see --multi-target \
  --algo ddpg_joint --max-episode-len 100 --max-iters 1000 --history-frame-len 5 \
  --segmentation-input color --depth-input \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

