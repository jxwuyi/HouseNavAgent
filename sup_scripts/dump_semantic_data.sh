#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 semantic_train.py --seed 0 \
    --data-dir ./_sup_data_/large --n-part 20 \
    --segmentation-input color --depth-input --resolution normal \
    --fixed-target kitchen \
    --train-gpu 0 \
    --batch-size 128 --grad-batch 1 --epochs 2 \
    --lrate 0.001 --weight-decay 0.00001 --grad-clip 10.0 --batch-norm \
    --optimizer adam \
    --save-dir ./_model_/semantic --log-dir ./log/semantic \
    --save-rate 1 --report-rate 1 \
    --eval-rate 1 --eval-dir ./_sup_data_/small --eval-n-part 1 \
    --only-data-loading --data-dump-dir ./
    #--eval-batch-size 100
    #--only-data-loading
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature
    # --logits-penalty 0.0001
