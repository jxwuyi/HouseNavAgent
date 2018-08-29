#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 data_gen.py --seed 10 --env-set train \
    --n-house 200 --n-partition 32 --n-proc 32 \
    --sample-size 200000 --t-max 50 \
    --include-object-target --fixed-target any-room \
    --hardness 0.6 --max-birthplace-steps 15 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input --resolution normal \
    --success-measure see-stop \
    --multi-target \ 
    --save-dir ./_sup_data_/large_full_obj --log-rate 500 \
    --max-expansion 100000 \
    --include-object-target \
    --include-mask-feature \
    --sanity-check
    #--no-outdoor-target \
    # --render-gpu 0,1

