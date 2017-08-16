
modelfile='./_model_/discrete/dqn/medium/eplen50_freq10_exp_low/DQN_DQNTrainer.pkl'
logfile='./log/eval/discrete/medium/dqn'

CUDA_VISIBLE_DEVICES=1 python eval.py --house 0 --seed 532 --hardness 0.5 \
  --algo dqn --max-episode-len 80 --max-iters 60 \
  --store-history --log-dir "$logfile"  --warmstart "$modelfile" --batch-norm
