#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python train.py --algo qac \
    --seed 0 --house 0 --update-freq 50 \
    --max-episode-len 50 --linear-reward --replay-buffer-size 50000 \
    --lrate 0.0001 --gamma 0.95 \
    --save-dir ./_model_/discrete/qac/medium/eplen50_freq50 \
    --log-dir ./log/discrete/qac/medium/eplen50_freq50 \
    --batch-size 256 --hardness 0.5 \
    --noise-scheduler low \
    --q-loss-coef 50 \
    --entropy-penalty 0.01 --batch-norm --no-debug  --weight-decay 0.0001 # --critic-penalty 0.0001
