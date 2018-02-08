# script for model trained on large set with object targets on semantic input
# test 
# a3c-seg-gate
CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --seed 0 --env-set test --house -50 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating \
    --include-object-target --only-eval-room-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/large/a3c_obj_seg_gate_test \
    --warmstart ./_model_/eval/large/a3c_obj_seg_gate/ZMQA3CTrainer_succ.pkl

# test 
# a3c-seg-nog
CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --seed 0 --env-set test --house -50 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --include-object-target --only-eval-room-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/large/a3c_obj_seg_nog_test \
    --warmstart ./_model_/eval/large/a3c_obj_seg_nog/ZMQA3CTrainer_succ.pkl


# train
# a3c-seg-gate
CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --seed 0 --env-set large --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating \
    --include-object-target --only-eval-room-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/large/a3c_obj_seg_gate_train \
    --warmstart ./_model_/eval/large/a3c_obj_seg_gate/ZMQA3CTrainer_succ.pkl

# train
# a3c-seg-nog
CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --seed 0 --env-set large --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --include-object-target --only-eval-room-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/large/a3c_obj_seg_nog_train \
    --warmstart ./_model_/eval/large/a3c_obj_seg_nog/ZMQA3CTrainer_succ.pkl



