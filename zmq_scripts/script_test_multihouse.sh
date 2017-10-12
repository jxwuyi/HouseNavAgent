#!/bin/bash


"""
Test A3C + Multi-House on 10house
>> fix target = kitchen
"""
modelfile='./_model_/zmq_a3c/10house/medium_color_see/delta_bn_bc64_tmax5/ZMQA3CTrainer_best.pkl'
logfile='./log/eval/zmq_a3c/full_test/easy_color_see/a3c_fixtar_10house_delta_bn_bc64_tmax5'

CUDA_VISIBLE_DEVICES=2 python3 eval.py --env-set test --house -50 --seed 0 --hardness 0.3 \
  --success-measure see --fixed-target kitchen \
  --algo a3c --max-episode-len 100 --max-iters 1000 \
  --segmentation-input color --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

