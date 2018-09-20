#!/bin/bash

#target='kitchen'
all_targets="outdoor living_room bedroom dining_room bathroom office garage"
large_samples='15000'
small_samples="1500"

for target in $all_targets
do

CUDA_VISIBLE_DEVICES=0,1,2 python3 data_gen_panoramic.py --seed 10 --env-set train \
    --n-house 200 --n-partition 10 --n-proc 10 \
    --sample-size $large_samples --neg-rate 1 \
    --fixed-target $target \
    --hardness 0.95 --max-birthplace-steps 50 \
    --segmentation-input color --depth-input --resolution normal \
    --render-gpu 0,1,2 \
    --save-dir ./_sup_data_/panoramic/large/$target --log-rate 20
    #--no-outdoor-target 
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature


CUDA_VISIBLE_DEVICES=2 python3 data_gen_panoramic.py --seed 10 --env-set small \
    --n-house 20 --n-partition 1 --n-proc 1 \
    --sample-size $small_samples --neg-rate 1 \
    --fixed-target $target \
    --hardness 0.8 --max-birthplace-steps 60 \
    --segmentation-input color --depth-input --resolution normal \
    --render-gpu 0 \
    --save-dir ./_sup_data_/panoramic/small/$target --log-rate 20
    #--no-outdoor-target 
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature

done
