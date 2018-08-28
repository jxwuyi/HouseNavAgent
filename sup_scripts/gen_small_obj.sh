#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 data_gen.py --seed 10 --env-set small \
    --n-house 20 --n-partition 20 --n-proc 20 \
    --sample-size 100000 --t-max 50 \
    --include-object-target --fixed-target any-room \
    --hardness 0.6 --max-birthplace-steps 15 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input --resolution normal \
    --success-measure see-stop \
    --multi-target \
    --save-dir ./_sup_data_/small_obj --log-rate 500 \
    --max-expansion 100000 \
    --include-mask-feature \
    --include-object-target \
    --sanity-check
    #--no-outdoor-target 
    # --render-gpu 0,1

