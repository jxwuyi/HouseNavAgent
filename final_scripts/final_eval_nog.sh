#
# a3c-vis-nog
CUDA_VISIBLE_DEVICES=0,2 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/final_full/a3c_vis_nog --warmstart ./_model_/cluster/full/a3c_vis_nog_ent005/ZMQA3CTrainer_succ.pkl

#
# a3c-seg-nog
CUDA_VISIBLE_DEVICES=0,2 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/final_full/a3c_seg_nog --warmstart ./_model_/cluster/full/a3c_segcolor_nog_ent005/ZMQA3CTrainer_succ.pkl




