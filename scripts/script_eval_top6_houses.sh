#!/bin/bash

# DQN
modelfile='./_model_/multi_house/linear_reward/6house_medium/dqn_eplen60_exp_exp/DQN_DQNTrainer_best.pkl'
logfile='./log/eval/multi_house/dqn/6house_medium/dqn_eplen60_exp_exp_best'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -6 --seed 621 --hardness 0.6 --algo dqn \
    --max-episode-len 70 --max-iters 120 --store-history \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

# early stopping, best model during training
modelfile='./_model_/multi_house/linear_reward/medium/dqn_eplen60_exp_exp/DQN_DQNTrainer.pkl'
logfile='./log/eval/multi_house/dqn/6house_medium/dqn_eplen60_exp_exp_final'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -6 --seed 622 --hardness 0.6 --algo dqn \
    --max-episode-len 70 --max-iters 120 --store-history \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
