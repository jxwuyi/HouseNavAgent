#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 semantic_train.py --seed 0 \
    --data-dir ./_sup_data_/large --n-part 20 \
    --segmentation-input color --depth-input --resolution normal \
    --fixed-target kitchen \
    --train-gpu 0 \
    --batch-size 256 --grad-batch 1 --epochs 50 \
    --lrate 0.001 --weight-decay 0.0001 --grad-clip 5.0 \
    --optimizer adam \
    --save-dir ./_model_/semantic/kitchen --log-dir ./log/semantic/kitchen \
    --save-rate 1 --report-rate 5 \
    --eval-rate 1 --eval-dir ./_sup_data_/small --eval-n-part 1 
    #--batch-norm
    #--eval-batch-size 100
    #--only-data-loading
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature
    # --logits-penalty 0.0001
