#!/bin/bash
"""
modelfile='./_model_/new_env/linear_reward/medium/bc_lr1e4_1e3_sftmx_c/DDPG_DDPGTrainer_best.pkl'
logfile='./log/eval/new_env/ddpg/noise/medium'

CUDA_VISIBLE_DEVICES=0 python eval.py --seed 71 --hardness 0.5 --algo ddpg --max-episode-len 70 --max-iters 50 --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm


modelfile='./_model_/new_env/linear_reward/hard/full_experiment_b/DDPG_DDPGTrainer_best.pkl'
logfile='./log/eval/new_env/ddpg/noise/hard/defaults'

CUDA_VISIBLE_DEVICES=0 python eval.py --seed 73 --hardness 0.95 --algo ddpg --max-episode-len 150 --max-iters 100 --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm


modelfile='./_model_/new_env/linear_reward/hard/long_experiment/DDPG_DDPGTrainer_best.pkl'
logfile='./log/eval/new_env/ddpg/noise/hard/long'

CUDA_VISIBLE_DEVICES=0 python eval.py --seed 77 --hardness 0.95 --algo ddpg --max-episode-len 200 --max-iters 100 --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
"""

"""
modelfile='./_model_/act3/house0/linear_reward/medium/bc_eplen50/DDPG_DDPGTrainer_best.pkl'
logfile='./log/eval/act3/house0/medium/bc_eplen50'

CUDA_VISIBLE_DEVICES=0 python eval.py --house 0 --seed 501 --hardness 0.3 --action-dim 3 \
  --algo ddpg --max-episode-len 100 --max-iters 70 \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
"""

"""
modelfile='./_model_/act3/house0/linear_reward/hard/exp_low_eplen100/DDPG_DDPGTrainer_best.pkl'
logfile='./log/eval/act3/house0/hard/exp_low_eplen100'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 0 --seed 521 --hardness 0.95 --action-dim 3 \
  --algo ddpg --max-episode-len 150 --max-iters 70 \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm

modelfile='./_model_/large_net/linear_reward/hard/exp_low_eplen100_freq10/DDPG_DDPGTrainer_best.pkl'
logfile='./log/eval/large_net/house0/hard/exp_low_eplen100_freq10'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 0 --seed 522 --hardness 0.95 \
  --algo ddpg --max-episode-len 150 --max-iters 70 \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
"""

modelfile='./_model_/joint/house0/linear_reward/medium/bc_eplen50_freq10/JointDDPG_JointDDPGTrainer.pkl'
logfile='./log/eval/joint/house0/medium/exp_low_joint_ddpg_coef50'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 0 --seed 532 --hardness 0.5 \
  --algo ddpg_joint --max-episode-len 100 --max-iters 70 \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
