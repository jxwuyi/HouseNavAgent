#!/bin/bash


"""
Test A3C + Multi-Target on 10house
>> fix target = kitchen
"""
modelfile='./_model_/zmq_a3c/10house_multitarget/medium_color/delta_bn_bc64_tmax5_gate/ZMQA3CTrainer_best.pkl'
logfile='./log/eval/zmq_a3c/full_test/easy_color_see/a3c_10house_delta_bn_bc64_tmax5_gate'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.3 \
  --success-measure see --multi-target --use-target-gating --fixed-target kitchen \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input color --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

