#!/bin/bash

"""
# ddpg_joint on 20-houses
modelfile='./_model_/multi_house/linear_reward/segjoint_20house_medium/ddpg_joint_100_exp_high_hist_3/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/segjoint_20house_medium/gating_coef100_eplen50_exp_high_hist_3_latest'

CUDA_VISIBLE_DEVICES=1 python eval.py --house -20 --seed 2021 --hardness 0.6 --algo ddpg_joint \
    --max-episode-len 70 --max-iters 400 --store-history --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
"""
# extra-house 21 easy
"""
# random
logfile='./log/eval/multi_house/nop/extra_house21_easy/gating_coef100_eplen50_exp_high_hist_3'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 21 --seed 2107 --hardness 0.3 --algo nop \
    --max-episode-len 80 --max-iters 300  --store-history --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 --batch-norm \
    --log-dir "$logfile"  #--warmstart "$modelfile"

"""



# ddpg_joint on 20-houses
modelfile='./_model_/multi_house/linear_reward/segjoint_20house_medium/ddpg_joint_100_exp_high_hist_3/JointDDPG_JointDDPGTrainer.pkl'
#modelfile='./_model_/multi_house/linear_reward/segjoint_15house_medium/ddpg_joint_100_exp_high_hist_3/JointDDPG_JointDDPGTrainer.pkl'
#modelfile='./_model_/multi_house/linear_reward/segjoint_10house_medium/ddpg_joint_100_exp_high_hist_3/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/multi_house/ddpg_joint/extra_house21_easy/gating_coef100_eplen50_exp_high_hist_3_latest'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 21 --seed 2107 --hardness 0.3 --algo ddpg_joint \
    --max-episode-len 80 --max-iters 500 --store-history --use-action-gating \
    --segmentation-input joint --resolution normal --history-frame-len 3 \
    --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
