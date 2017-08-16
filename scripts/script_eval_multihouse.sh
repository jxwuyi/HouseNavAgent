#!/bin/bash

"""
# DQN
modelfile='./_model_/multi_house/linear_reward/medium/dqn_eplen60_exp_exp/DQN_DQNTrainer_best.pkl'
logfile='./log/eval/multi_house/dqn/medium/dqn_eplen60_exp_exp_best'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -3 --seed 421 --hardness 0.6 --algo dqn \
    --max-episode-len 70 --max-iters 70 --store-history \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

modelfile='./_model_/multi_house/linear_reward/medium/dqn_eplen60_exp_exp/DQN_DQNTrainer.pkl'
logfile='./log/eval/multi_house/dqn/medium/dqn_eplen60_exp_exp_final'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -3 --seed 422 --hardness 0.6 --algo dqn \
    --max-episode-len 70 --max-iters 70 --store-history \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
"""

"""
# DDPG
modelfile='./_model_/multi_house/linear_reward/medium/ddpg_joint_150_exp_high/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/medium/ddpg_coef150_eplen50_exp_high'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -3 --seed 423 --hardness 0.6 --algo ddpg_joint \
    --max-episode-len 70 --max-iters 70 --store-history \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
"""

# DDPG + ActionGating
"""
modelfile='./_model_/multi_house/linear_reward/6house_medium/gating_ddpg_joint_100_exp_high/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/6house_medium/gating_coef100_eplen50_exp_high'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -6 --seed 625 --hardness 0.6 --algo ddpg_joint \
    --max-episode-len 70 --max-iters 120 --store-history --use-action-gating \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
"""

# 10 houses
modelfile='./_model_/multi_house/linear_reward/10house_medium/gating_ddpg_joint_100_exp_high/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/10house_medium/gating_coef100_eplen50_exp_high'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -10 --seed 1023 --hardness 0.6 --algo ddpg_joint \
    --max-episode-len 70 --max-iters 160 --store-history --use-action-gating \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

"""
# 3 house hard
modelfile='./_model_/multi_house/linear_reward/hard/gating_ddpg_joint_100_exp_high/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/hard/gating_coef100_eplen50_exp_high'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -3 --seed 320 --hardness 0.95 --algo ddpg_joint \
    --max-episode-len 100 --max-iters 70 --store-history --use-action-gating \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
"""


# dist sampling
modelfile='./_model_/multi_house/linear_reward/hard/distsample_gating_ddpg_joint_100_exp_high/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/hard/distsample_gating_coef100_eplen50_exp_high'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -3 --seed 332 --hardness 0.95 --algo ddpg_joint \
    --max-episode-len 100 --max-iters 70 --store-history --use-action-gating \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
