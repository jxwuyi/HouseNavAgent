#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 data_gen.py --seed 10 --env-set small \
    --n-house 5 --n-partition 2 --n-proc 2 \
    --sample-size 100 --t-max 30 \
    --include-object-target --fixed-target any-room \
    --hardness 0.6 --max-birthplace-steps 10 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input --resolution normal \
    --success-measure see-stop \
    --multi-target \
    --save-dir ./_sup_data_/tiny --log-rate 20 \
    --max-expansion 100000 \
    --sanity-check
    #--no-outdoor-target 
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature

