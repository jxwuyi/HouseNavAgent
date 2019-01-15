#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 data_gen.py --seed 10 --env-set small \
    --n-house 20 --n-partition 5 --n-proc 5 \
    --sample-size 2000 --t-max 30 \
    --include-object-target --fixed-target outdoor \
    --hardness 0.6 --max-birthplace-steps 15 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input --resolution normal \
    --success-measure see-stop \
    --multi-target \
    --save-dir ./_sup_data_/small_outdoor --log-rate 500 \
    --max-expansion 100000 \
    --sanity-check
    #--include-mask-feature \
    #--no-outdoor-target 
    # --render-gpu 0,1
    # --include-object-target

