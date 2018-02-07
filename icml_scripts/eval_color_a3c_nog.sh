# train
# a3c-vis-nog
CUDA_VISIBLE_DEVICES=1 python3 eval.py --seed 0 --env-set small --house -20 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/color/a3c_vis_nog_train \
    --warmstart ./_model_/eval/color/a3c_vis_nog/ZMQA3CTrainer_best.pkl

# test 
# a3c-vis-nog
CUDA_VISIBLE_DEVICES=1 python3 eval.py --seed 0 --env-set test --house -50 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/color/a3c_vis_nog_test \
    --warmstart ./_model_/eval/color/a3c_vis_nog/ZMQA3CTrainer_best.pkl


