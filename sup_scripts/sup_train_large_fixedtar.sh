#!/bin/bash
MODEL_NAME='train_kitchen_t20_ent5e1_lgt1e1'
CUDA_VISIBLE_DEVICES=1 python3 sup_train.py --seed 0 \
    --data-dir ./_sup_data_/large --n-part 20 \
    --segmentation-input color --depth-input --resolution normal \
    --multi-target --fixed-target kitchen \
    --train-gpu 0 \
    --t-max 20 --batch-size 100 --grad-batch 1 --epochs 15 \
    --lrate 0.001 --weight-decay 0.0001 --grad-clip 1.0 --batch-norm \
    --optimizer adam --entropy-penalty 0.5 --logits-penalty 0.1 \
    --use-target-gating \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
    --save-dir ./_model_/supervise/$MODEL_NAME --log-dir ./log/supervise/$MODEL_NAME \
    --save-rate 1 --report-rate 10 \
    --eval-rate 1 --eval-dir ./_sup_data_/small --eval-n-part 1 
    #--eval-batch-size 100
    #--only-data-loading
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature
    # --logits-penalty 0.0001
    # --random-data-clip 
