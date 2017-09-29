#!/bin/bash
modelfile='./_model_/joint/house0/linear_reward/medium/bc_eplen50_freq10/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/joint/house0/medium/exp_low_joint_ddpg_coef50'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 0 --seed 532 --hardness 0.4 \
  --algo a3c --max-episode-len 100 --max-iters 100 \
  --segmentation-input joint --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

