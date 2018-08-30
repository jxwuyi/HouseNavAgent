#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 sup_train.py --seed 0 \
    --data-dir ./_sup_data_/large --n-part 20 --random-data-clip \
    --segmentation-input color --depth-input --resolution normal \
    --multi-target --fixed-target any-room \
    --train-gpu 0 \
    --t-max 20 --batch-size 100 --grad-batch 1 --epochs 100 \
    --lrate 0.001 --weight-decay 0.00001 --grad-clip 5.0 --batch-norm \
    --optimizer adam --entropy-penalty 0.001 \
    --use-target-gating \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --save-dir ./_model_/supervise/train_large --log-dir ./log/supervise/train_large \
    --save-rate 1 --report-rate 10 \
    --eval-rate 1 --eval-dir ./_sup_data_/small --eval-n-part 1 
    #--eval-batch-size 100
    #--only-data-loading
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature
    # --logits-penalty 0.0001
