#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 sup_train.py --seed 0\
    --data-dir ./_sup_data/large --n-part 2 \
    --segmentation-input color --depth-input --resolution normal \
    --multi-target --fixed-target any-room \
    --train-gpu 0 \
    --t-max 30 --batch-size 64 --grad-batch 1 --epochs 10 \
    --lrate 0.001 --weight-decay 0.00001 --grad-clip 2.0 --batch-norm \
    --optimizer adam --entropy-penalty 0.001 --logits-penalty 0.0001 \
    --use-target-gating \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --save-dir ./_model_/supervise --log-dir ./log/supervise \
    --save-rate 1 --report-rate 5 \
    --eval-rate 1 --eval-dir ./_sup_data/small --eval-n-part 1 --eval-batch-size 64
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature
    # --logits-penalty
    # --only-data-loading
