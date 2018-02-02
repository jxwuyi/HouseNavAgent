
"""#
# nop-discrete
CUDA_VISIBLE_DEVICES=1,2 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating \
    --algo nop --max-episode-len 100 --max-iters 5000 --greedy-execution\
    --segmentation-input color \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/final_full/nop_disc


"""

# nop-continuous
CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating \
    --algo nop --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/eval/final_full/nop_cont

