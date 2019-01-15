#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python3 data_gen.py --seed 10 --env-set train \
    --n-house 200 --n-partition 20 --n-proc 10 \
    --sample-size 10000 --t-max 25 \
    --include-object-target --fixed-target outdoor \
    --hardness 0.6 --max-birthplace-steps 15 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input --resolution normal \
    --success-measure see-stop \
    --multi-target \
    --save-dir ./_sup_data_/outdoor --log-rate 500 \
    --max-expansion 50000 \
    --sanity-check
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature

