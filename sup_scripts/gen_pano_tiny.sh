#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 data_gen.py --seed 10 --env-set small \
    --n-house 5 --n-partition 2 --n-proc 2 \
    --sample-size 100 --neg-rate 1 \
    --fixed-target kitchen \
    --hardness 0.6 --max-birthplace-steps 40 \
    --segmentation-input color --depth-input --resolution normal \
    --multi-target \
    --render_gpu 0 \
    --save-dir ./_sup_data_/panoramic/tiny --log-rate 20
    #--no-outdoor-target 
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature

