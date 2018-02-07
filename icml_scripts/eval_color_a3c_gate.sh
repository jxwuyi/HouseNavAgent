# train
# a3c-vis-gate
CUDA_VISIBLE_DEVICES=0 python3 eval.py --seed 0 --env-set small --house -20 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/color/a3c_vis_gate_train \
    --warmstart ./_model_/eval/color/a3c_vis_gate/ZMQA3CTrainer_best.pkl

# test 
# a3c-vis-gate
CUDA_VISIBLE_DEVICES=0 python3 eval.py --seed 0 --env-set test --house -50 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/color/a3c_vis_gate_test \
    --warmstart ./_model_/eval/color/a3c_vis_gate/ZMQA3CTrainer_best.pkl


