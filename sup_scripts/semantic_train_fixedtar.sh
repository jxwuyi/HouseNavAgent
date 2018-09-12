#!/bin/bash
target="outdoor"
data_dir="outdoor"
test_data_dir="small_outdoor"
test_part=5
#data_dir="large"
CUDA_VISIBLE_DEVICES=1 python3 semantic_train.py --seed 0 \
    --data-dir ./_sup_data_/$data_dir --n-part 20 \
    --segmentation-input color --depth-input --resolution normal \
    --fixed-target $target \
    --train-gpu 0 \
    --batch-size 256 --grad-batch 1 --epochs 50 \
    --lrate 0.0005 --weight-decay 0.00001 --grad-clip 5.0 \
    --optimizer adam --batch-norm \
    --save-dir ./_model_/semantic/$target --log-dir ./log/semantic/$target \
    --save-rate 1 --report-rate 20 \
    --eval-rate 1 --eval-dir ./_sup_data_/$test_data_dir --eval-n-part $test_part 
    #--batch-norm
    #--eval-batch-size 100
    #--only-data-loading
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature
    # --logits-penalty 0.0001
