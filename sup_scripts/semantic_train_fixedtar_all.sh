#!/bin/bash

#all_targets="outdoor kitchen dining_room living_room bathroom bedroom office garage"

current_targets="living_room bathroom bedroom office garage"


for target in $current_targets
do
CUDA_VISIBLE_DEVICES=0 python3 semantic_train.py --seed 0 \
    --data-dir ./_sup_data_/large --n-part 20 \
    --segmentation-input color --depth-input --resolution normal \
    --fixed-target $target \
    --train-gpu 0 \
    --batch-size 256 --grad-batch 1 --epochs 50 \
    --lrate 0.0005 --weight-decay 0.00001 --grad-clip 5.0 \
    --optimizer adam --batch-norm \
    --save-dir ./_model_/semantic/$target --log-dir ./log/semantic/$target \
    --save-rate 1 --report-rate 20 \
    --eval-rate 1 --eval-dir ./_sup_data_/small --eval-n-part 1 
    #--batch-norm
    #--eval-batch-size 100
    #--only-data-loading
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature
    # --logits-penalty 0.0001
    #read -p "Press any key to continue..." -n1 -s
done
