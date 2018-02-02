#!/bin/bash


"""
Test A3C + Multi-Target + Segcolor + Tmax30
"""

modelfile='./_model_/cluster/full/a3c_segcolor_gate_ent005_sd17/ZMQA3CTrainer_succ.pkl'
logfile='./log/eval/cluster/full/a3c_segcolor_gate_ent005_sd17_succ'

#modelfile='./_model_/new/zmq_a3c/small/hard_color/bn_bc64_gate_tmax20/ZMQA3CTrainer_succ.pkl'
#logfile='./log/eval/new/zmq_a3c/small/hard_color/small_bc64_gate_tmax20'
#modelfile='./_model_/cluster/small/a3c_segcolor_gate_new/ZMQA3CTrainer_succ.pkl'
#modelfile='./_model_/new/zmq_a3c/small/hard_color/bn_bc64_gate_tmax30/ZMQA3CTrainer_succ.pkl'
#logfile='./log/eval/cluster/small/small_a3c_segcolor_gate_local'
#logfile='./log/eval/cluster/small/small_a3c_segcolor_gate'
#modelfile='./_model_/cluster/full/a3c_segcolor_gate_ent005/ZMQA3CTrainer_succ.pkl'
#logfile='./log/eval/cluster/full/full_a3c_segcolor_gate_ent005'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target --use-target-gating \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input color --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

