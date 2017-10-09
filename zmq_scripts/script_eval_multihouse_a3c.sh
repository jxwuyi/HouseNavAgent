#!/bin/bash
modelfile='./_model_/zmq_a3c/10house/medium_color_see/delta_bn_bc64_tmax5/ZMQA3CTrainer_best.pkl'
logfile='./log/eval/zmq_a3c/house_16/easy_color_see/delta_bn_bc64_tmax5'

CUDA_VISIBLE_DEVICES=1 python3 eval.py --house 16 --seed 0 --hardness 0.3 \
  --success-measure see \
  --algo a3c --max-episode-len 100 --max-iters 100 \
  --segmentation-input color --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

