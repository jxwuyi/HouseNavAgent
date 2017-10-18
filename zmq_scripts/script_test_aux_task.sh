#!/bin/bash


"""
Test A3C + Multi-Target + Aux Task on 20house
"""
modelfile='./_model_/zmq_a3c/aux_task_20house/hard_color/delta_bn_bc64_tmax5_gate_auxc1/ZMQAuxTaskA3CTrainer_best.pkl'
logfile='./log/eval/zmq_a3c/full_test/hard_color_see/a3c_multitarget_gate_aux_task_c1'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.95 \
  --success-measure see --multi-target --use-target-gating \
  --auxiliary-task \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input color --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

