#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python train.py --seed 0 --update-freq 25 --linear-reward --lrate 0.0001 --target-net-update-rate 0.001 --gamma 0.95 --save-dir ./_model_/linear_reward/medium/eta1e3_lr1e4 --log-dir ./log/linear_reward/medium/eta1e3_lr1e4 --batch-size 256 --hardness 0.5
