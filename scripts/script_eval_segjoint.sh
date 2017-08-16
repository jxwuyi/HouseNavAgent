#!/bin/bash

"""

# 3 houses - hard
modelfile='./_model_/multi_house/linear_reward/segjoint_medium/ddpg_joint_100_exp_high_hist_3/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/segjoint_hard/gating_coef100_eplen50_exp_high_hist_3'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -3 --seed 391 --hardness 0.95 --algo ddpg_joint \
    --max-episode-len 100 --max-iters 60 --store-history --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm


# 10 houses - medium
modelfile='./_model_/multi_house/linear_reward/segjoint_10house_medium/ddpg_joint_100_exp_high_hist_3/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/segjoint_10house_medium/gating_coef100_eplen50_exp_high_hist_3'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -10 --seed 1013 --hardness 0.6 --algo ddpg_joint \
    --max-episode-len 70 --max-iters 120 --store-history --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm


# 15 houses - medium
modelfile='./_model_/multi_house/linear_reward/segjoint_15house_medium/ddpg_joint_100_exp_high_hist_3/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/segjoint_15house_medium/gating_coef100_eplen50_exp_high_hist_3'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -15 --seed 1501 --hardness 0.6 --algo ddpg_joint \
    --max-episode-len 70 --max-iters 180 --store-history --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

"""


# extra-house 16 easy
"""
# random
logfile='./log/eval/multi_house/nop/extra_house16_easy/gating_coef100_eplen50_exp_high_hist_3'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 16 --seed 1501 --hardness 0.3 --algo nop \
    --max-episode-len 70 --max-iters 50  --store-history --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 --batch-norm \
    --log-dir "$logfile"  #--warmstart "$modelfile"
"""

# ddpg_joint on 15-houses
modelfile='./_model_/multi_house/linear_reward/segjoint_15house_medium/ddpg_joint_100_exp_high_hist_3/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/extra_house16_medium/gating_coef100_eplen50_exp_high_hist_3'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 16 --seed 1501 --hardness 0.6 --algo ddpg_joint \
    --max-episode-len 70 --max-iters 50 --store-history --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
