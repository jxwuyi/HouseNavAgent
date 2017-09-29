#!/bin/bash
modelfile='./_model_/zmq_a3c/1house/medium/bn_bc64_tmax5/ZMQA3CTrainer.pkl'
logfile='./log/eval/zmq_a3c/1house/medium/bn_bc64_tmax5'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 0 --seed 532 --hardness 0.4 \
  --algo a3c --max-episode-len 100 --max-iters 100 \
  --segmentation-input joint --depth-input \
  --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

